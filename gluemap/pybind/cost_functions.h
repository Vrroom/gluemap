#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "vendor/colmap/estimators/cost_functions/reprojection_error.h"
#include "vendor/colmap/estimators/cost_functions/utils.h"

// ----------------------------------------
// RotationGeodesicError
// ----------------------------------------
// Computes the geodesic error between rotation quaternions.
struct RotationGeodesicError
    : public colmap::AutoDiffCostFunctor<RotationGeodesicError, 3, 4, 4> {
public:
  explicit RotationGeodesicError(const Eigen::Vector4d &j_q_i)
      : j_q_i_(j_q_i) {}

  template <typename T>
  bool operator()(const T *const i_q_w, const T *const j_q_w,
                  T *residuals_ptr) const {
    const T w_q_j[4] = {j_q_w[0], -j_q_w[1], -j_q_w[2], -j_q_w[3]};

    T tmp_i_q_j[4];
    ceres::QuaternionProduct(i_q_w, w_q_j, tmp_i_q_j);

    T q_res[4];
    const Eigen::Matrix<T, 4, 1> j_q_i = j_q_i_.cast<T>();
    ceres::QuaternionProduct(j_q_i.data(), tmp_i_q_j, q_res);

    ceres::QuaternionToAngleAxis(q_res, residuals_ptr);

    return true;
  }

private:
  const Eigen::Vector4d j_q_i_;
};

// ----------------------------------------
// PairwiseDirectionError
// ----------------------------------------
// Computes the error between a translation direction and the direction formed
// from two positions such that t_ij - scale * (c_j - c_i) is minimized.
struct PairwiseDirectionError
    : public colmap::AutoDiffCostFunctor<PairwiseDirectionError, 3, 3, 3, 1> {
  PairwiseDirectionError(const Eigen::Vector3d &translation_obs)
      : translation_obs_(translation_obs) {}

  template <typename T>
  bool operator()(const T *position1, const T *position2, const T *scale,
                  T *residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        translation_obs_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1));
    return true;
  }

private:
  const Eigen::Vector3d translation_obs_;
};

// ----------------------------------------
// PairwiseDirectionErrorWithOffset
// ----------------------------------------
// Like PairwiseDirectionError, but with a known constant offset added to the
// position difference: t_ij - scale * (c_j - c_i + offset) is minimized.
struct PairwiseDirectionErrorWithOffset
    : public colmap::AutoDiffCostFunctor<PairwiseDirectionErrorWithOffset, 3, 3,
                                         3, 1> {
  PairwiseDirectionErrorWithOffset(const Eigen::Vector3d &translation_obs,
                                   const Eigen::Vector3d &offset)
      : translation_obs_(translation_obs), offset_(offset) {}

  template <typename T>
  bool operator()(const T *position1, const T *position2, const T *scale,
                  T *residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        translation_obs_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1) +
                    offset_.cast<T>());
    return true;
  }

private:
  const Eigen::Vector3d translation_obs_;
  const Eigen::Vector3d offset_;
};

// ----------------------------------------
// ScaleOnlyDirectionError
// ----------------------------------------
// The scale-only residual for an intra-rig edge, where the two camera centers
// share a reference so their difference is the known offset alone:
// t_ij - scale * offset is minimized. Only the ministar scale is a parameter.
struct ScaleOnlyDirectionError
    : public colmap::AutoDiffCostFunctor<ScaleOnlyDirectionError, 3, 1> {
  ScaleOnlyDirectionError(const Eigen::Vector3d &translation_obs,
                          const Eigen::Vector3d &offset)
      : translation_obs_(translation_obs), offset_(offset) {}

  template <typename T>
  bool operator()(const T *scale, T *residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec = translation_obs_.cast<T>() - scale[0] * offset_.cast<T>();
    return true;
  }

private:
  const Eigen::Vector3d translation_obs_;
  const Eigen::Vector3d offset_;
};

// ----------------------------------------
// RelativePosePriorError
// ----------------------------------------
// Soft prior on the relative pose between two poses a_from_world and
// b_from_world. Penalizes the predicted b_from_a against a known prior with
// separate rotation and translation weights. For soft rig constraints the
// fixed sensor_from_rig transforms are folded into the prior on the Python
// side, so both parameter blocks are frame poses (rig_from_world).
struct RelativePosePriorError
    : public colmap::AutoDiffCostFunctor<RelativePosePriorError, 6, 7, 7> {
  RelativePosePriorError(const Eigen::Matrix3d &b_R_a,
                         const Eigen::Vector3d &b_t_a, double w_rot,
                         double w_trans)
      : b_q_a_(Eigen::Quaterniond(b_R_a)),
        b_t_a_(b_t_a),
        w_rot_(w_rot),
        w_trans_(w_trans) {}

  template <typename T>
  bool operator()(const T *a_from_world, const T *b_from_world,
                  T *residuals) const {
    const Eigen::Quaternion<T> q_a = colmap::EigenQuaternionMap<T>(a_from_world);
    const Eigen::Matrix<T, 3, 1> t_a =
        colmap::EigenVector3Map<T>(a_from_world + 4);
    const Eigen::Quaternion<T> q_b = colmap::EigenQuaternionMap<T>(b_from_world);
    const Eigen::Matrix<T, 3, 1> t_b =
        colmap::EigenVector3Map<T>(b_from_world + 4);

    // Predicted b_from_a = b_from_world * world_from_a.
    const Eigen::Quaternion<T> q_ba = q_b * q_a.conjugate();
    const Eigen::Matrix<T, 3, 1> t_ba = t_b - (q_ba * t_a);

    const Eigen::Quaternion<T> q_err = b_q_a_.conjugate().cast<T>() * q_ba;
    Eigen::Map<Eigen::Matrix<T, 6, 1>> r(residuals);
    r.template head<3>() = (T(2) * T(w_rot_)) * q_err.vec();
    r.template tail<3>() = T(w_trans_) * (t_ba - b_t_a_.cast<T>());
    return true;
  }

private:
  const Eigen::Quaterniond b_q_a_;
  const Eigen::Vector3d b_t_a_;
  const double w_rot_;
  const double w_trans_;
};

// ----------------------------------------
// ReprojErrorCostWithNegativeDepthFunctor
// ----------------------------------------
// Standard bundle adjustment cost function for variable
// camera pose, calibration, and point parameters.
// This version handles negative depth (points behind camera).
template <typename CameraModel>
class ReprojErrorCostWithNegativeDepthFunctor
    : public colmap::AutoDiffCostFunctor<
          ReprojErrorCostWithNegativeDepthFunctor<CameraModel>, 2, 3, 7,
          CameraModel::num_params> {
public:
  explicit ReprojErrorCostWithNegativeDepthFunctor(
      const Eigen::Vector2d &point2D)
      : point2D_(point2D) {}

  template <typename T>
  bool operator()(const T *const point3D, const T *const cam_from_world,
                  const T *const camera_params, T *residuals) const {
    Eigen::Matrix<T, 3, 1> point3D_in_cam =
        colmap::EigenQuaternionMap<T>(cam_from_world) *
            colmap::EigenVector3Map<T>(point3D) +
        colmap::EigenVector3Map<T>(cam_from_world + 4);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals_vec(residuals);

    // Always negate the point for negative depth projection
    point3D_in_cam = -point3D_in_cam;
    if (CameraModel::ImgFromCam(camera_params, point3D_in_cam[0],
                                point3D_in_cam[1], point3D_in_cam[2],
                                &residuals[0], &residuals[1])) {
      residuals_vec -= point2D_.cast<T>();
    } else {
      residuals_vec.setZero();
    }
    return true;
  }

private:
  const Eigen::Vector2d point2D_;
};

// ----------------------------------------
// RigReprojErrorCostWithNegativeDepthFunctor
// ----------------------------------------
// Like ReprojErrorCostWithNegativeDepthFunctor, but the optimized pose is the
// frame's rig_from_world; a fixed sensor_from_rig is composed on top so the
// point lands in the member sensor's frame. Handles negative depth.
template <typename CameraModel>
class RigReprojErrorCostWithNegativeDepthFunctor
    : public colmap::AutoDiffCostFunctor<
          RigReprojErrorCostWithNegativeDepthFunctor<CameraModel>, 2, 3, 7,
          CameraModel::num_params> {
public:
  RigReprojErrorCostWithNegativeDepthFunctor(
      const Eigen::Vector2d &point2D, const Eigen::Matrix3d &sensor_from_rig_R,
      const Eigen::Vector3d &sensor_from_rig_t)
      : point2D_(point2D),
        q_sr_(Eigen::Quaterniond(sensor_from_rig_R)),
        t_sr_(sensor_from_rig_t) {}

  template <typename T>
  bool operator()(const T *const point3D, const T *const rig_from_world,
                  const T *const camera_params, T *residuals) const {
    // cam_from_world = sensor_from_rig * rig_from_world, applied to the point
    // in two steps: first into the rig frame, then the fixed sensor_from_rig.
    const Eigen::Matrix<T, 3, 1> point3D_in_rig =
        colmap::EigenQuaternionMap<T>(rig_from_world) *
            colmap::EigenVector3Map<T>(point3D) +
        colmap::EigenVector3Map<T>(rig_from_world + 4);
    Eigen::Matrix<T, 3, 1> point3D_in_cam =
        q_sr_.cast<T>() * point3D_in_rig + t_sr_.cast<T>();
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals_vec(residuals);

    // Always negate the point for negative depth projection
    point3D_in_cam = -point3D_in_cam;
    if (CameraModel::ImgFromCam(camera_params, point3D_in_cam[0],
                                point3D_in_cam[1], point3D_in_cam[2],
                                &residuals[0], &residuals[1])) {
      residuals_vec -= point2D_.cast<T>();
    } else {
      residuals_vec.setZero();
    }
    return true;
  }

private:
  const Eigen::Vector2d point2D_;
  const Eigen::Quaterniond q_sr_;
  const Eigen::Vector3d t_sr_;
};

// ----------------------------------------
// RigReprojErrorCostFunctor
// ----------------------------------------
// Positive-depth rig reprojection: the optimized pose is rig_from_world,
// composed with a fixed sensor_from_rig. Same as
// RigReprojErrorCostWithNegativeDepthFunctor without the point negation.
template <typename CameraModel>
class RigReprojErrorCostFunctor
    : public colmap::AutoDiffCostFunctor<RigReprojErrorCostFunctor<CameraModel>,
                                         2, 3, 7, CameraModel::num_params> {
public:
  RigReprojErrorCostFunctor(const Eigen::Vector2d &point2D,
                            const Eigen::Matrix3d &sensor_from_rig_R,
                            const Eigen::Vector3d &sensor_from_rig_t)
      : point2D_(point2D),
        q_sr_(Eigen::Quaterniond(sensor_from_rig_R)),
        t_sr_(sensor_from_rig_t) {}

  template <typename T>
  bool operator()(const T *const point3D, const T *const rig_from_world,
                  const T *const camera_params, T *residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_rig =
        colmap::EigenQuaternionMap<T>(rig_from_world) *
            colmap::EigenVector3Map<T>(point3D) +
        colmap::EigenVector3Map<T>(rig_from_world + 4);
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        q_sr_.cast<T>() * point3D_in_rig + t_sr_.cast<T>();
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals_vec(residuals);

    if (CameraModel::ImgFromCam(camera_params, point3D_in_cam[0],
                                point3D_in_cam[1], point3D_in_cam[2],
                                &residuals[0], &residuals[1])) {
      residuals_vec -= point2D_.cast<T>();
    } else {
      residuals_vec.setZero();
    }
    return true;
  }

private:
  const Eigen::Vector2d point2D_;
  const Eigen::Quaterniond q_sr_;
  const Eigen::Vector3d t_sr_;
};
