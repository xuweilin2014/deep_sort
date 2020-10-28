# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    # 是一个多目标 tracker，保存了很多个 track 轨迹
    # 负责调用卡尔曼滤波来预测 track 的新状态 + 进行匹配工作 + 初始化第一帧
    # Tracker 调用 update 或 predict 的时候，其中的每个 track 也会各自调用自己的 update 或 predict
    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):

        #  metric 是一个类，用于计算距离(余弦距离或马氏距离)
        self.metric = metric

        # 最大 iou，iou 匹配的时候使用
        self.max_iou_distance = max_iou_distance

        # max_age 直接指定级联匹配的 cascade_depth 参数的值
        self.max_age = max_age

        # n_init 代表需要 n_init 次数的 update 才会将 track 状态设置为confirmed
        self.n_init = n_init

        # 卡尔曼滤波器
        self.kf = kalman_filter.KalmanFilter()

        # 保存一系列轨迹
        self.tracks = []

        # 下一个分配的轨迹 id
        self._next_id = 1

    def predict(self):
        """
        Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        # 遍历每一个 track 都进行一次 predict
        for track in self.tracks:
            track.predict(self.kf)

    # 进行测量的更新和轨迹的管理
    def update(self, detections):

        # 进行级联匹配
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        # 针对匹配上的结果
        for track_idx, detection_idx in matches:
            # track 更新对应的 detection，主要包括以下更新操作
            # 1.更新卡尔曼滤波的一系列运动变量，命中次数 hits 以及 time_since_update
            # 2.detection 的深度特征保存到这个 track 的特征集合中
            # 3.如果已经连续匹配上 n_init 帧，那么就把 track 的状态由 tentative 转变为 confirmed
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        # 针对未匹配的 track，调用 mark_missed 进行处理
        # 若 track 处于未确定状态，则标记为 deleted 状态
        # 若 track 失配的次数大于 max_age，那么也标记为 deleted 状态
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 针对未匹配上任何 track 的 detection，那么就为其初始化一个新的 track
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # 得到新的 tracks 列表，保存的是标记为 confirmed 和 tentative 的 track
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []

        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        # 主要功能是进行匹配，找到匹配的，未匹配的部分
        def gated_metric(tracks, dets, track_indices, detection_indices):
            # 功能：用于计算 track 和 detection 之间的距离，代价函数需要使用在 KM 算法之前
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # 通过最近邻计算出代价矩阵
            cost_matrix = self.metric.distance(features, targets)
            # 计算马氏距离，得到新的代价矩阵
            cost_matrix = linear_assignment.gate_cost_matrix(self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)

            return cost_matrix

        # 将轨迹的状态划分为 confirmed_tracks 和 unconfirmed_tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 进行级联匹配，得到匹配
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # 将所有状态为未确定态的轨迹和刚刚没有匹配上的轨迹组合为 iou_track_candidates
        # tracks[k].time_since_update == 1 表示刚刚没有匹配上
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        # tracks[k].time_since_update != 1 表示已经很久没有匹配上
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]

        # 进行 IOU 匹配，也就是说对级联匹配中没有匹配上的目标再进行 IOU 匹配
        # 虽然和级联匹配中使用的都是 min_cost_matching 作为核心，这里使用的 metric 是 iou cost 和以上不同
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature))
        self._next_id += 1
