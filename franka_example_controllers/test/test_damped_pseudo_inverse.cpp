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

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "franka_example_controllers/damped_pseudo_inverse.hpp"

namespace {

using Jacobian = Eigen::Matrix<double, 6, 7>;

TEST(DampedPseudoInverse, MatchesSingularValueDecomposition) {
  const Jacobian jacobian = Jacobian::Random();
  constexpr double kDamping = 0.2;

  const auto actual =
      franka_example_controllers::dampedPseudoInverseOfTranspose<6, 7>(jacobian, kDamping);

  const Eigen::Matrix<double, 7, 6> jacobian_transpose = jacobian.transpose();
  const Eigen::JacobiSVD<Eigen::Matrix<double, 7, 6>> svd(
      jacobian_transpose, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, 7, 6> damped_singular_values = Eigen::Matrix<double, 7, 6>::Zero();
  for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i) {
    const double singular_value = svd.singularValues()(i);
    damped_singular_values(i, i) =
        singular_value / (singular_value * singular_value + kDamping * kDamping);
  }
  const Jacobian expected =
      svd.matrixV() * damped_singular_values.transpose() * svd.matrixU().transpose();

  EXPECT_TRUE(actual.isApprox(expected, 1e-12));
}

TEST(DampedPseudoInverse, RemainsFiniteForSingularJacobian) {
  Jacobian jacobian = Jacobian::Zero();
  jacobian.topLeftCorner<3, 3>().setIdentity();

  const auto result =
      franka_example_controllers::dampedPseudoInverseOfTranspose<6, 7>(jacobian, 0.2);

  EXPECT_TRUE(result.allFinite());
}

}  // namespace
