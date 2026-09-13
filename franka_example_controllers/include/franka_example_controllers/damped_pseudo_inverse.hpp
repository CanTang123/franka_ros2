// Copyright (c) 2026 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace franka_example_controllers {

/**
 * Compute the damped pseudo-inverse of a Jacobian transpose without heap allocation.
 *
 * For a wide Jacobian J, this evaluates
 *   (J^T)^+ = (J J^T + damping^2 I)^-1 J
 * using fixed-size matrices and an LDLT solve. This is the form needed by the
 * Cartesian impedance controller's nullspace projector.
 */
template <int Rows, int Cols>
Eigen::Matrix<double, Rows, Cols> dampedPseudoInverseOfTranspose(
    const Eigen::Matrix<double, Rows, Cols>& jacobian,
    double damping) {
  static_assert(Rows <= Cols, "Expected a square or wide Jacobian");

  using SquareMatrix = Eigen::Matrix<double, Rows, Rows>;
  const SquareMatrix regularized =
      jacobian * jacobian.transpose() + damping * damping * SquareMatrix::Identity();
  return regularized.ldlt().solve(jacobian);
}

}  // namespace franka_example_controllers
