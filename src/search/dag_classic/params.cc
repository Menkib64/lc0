/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2025 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "search/dag_classic/params.h"

#include <cmath>

namespace lczero {
namespace dag_classic {

const OptionId SearchParams::kUseUncertaintyWeightingId{
    {.long_flag = "use-uncertainty-weighting",
     .uci_option = "UseUncertaintyWeighting",
     .help_text = "Enable uncertainty weighting in MCTS."}};
const OptionId SearchParams::kUncertaintyWeightingMidPointId{
    {.long_flag = "uncertainty-weighting-mid-point",
     .uci_option = "UncertaintyWeightingMidPoint",
     .help_text = "The mid point of logistic function."}};
const OptionId SearchParams::kUncertaintyWeightingMinusExponentId{
    {.long_flag = "uncertainty-weighting-minus-exponent",
     .uci_option = "UncertaintyWeightingMinusExponent",
     .help_text = "Minus exponent in the logistic function."}};
const OptionId SearchParams::kUncertaintyWeightingScaleId{
    {.long_flag = "uncertainty-weighting-scale",
     .uci_option = "UncertaintyWeightingScale",
     .help_text = "Numerator in the logistic function."}};
const OptionId SearchParams::kUncertaintyWeightingBaseId{
    {.long_flag = "uncertainty-weighting-base",
     .uci_option = "UncertaintyWeightingBase",
     .help_text = "Base value in the logistic function."}};
const OptionId SearchParams::kUncertaintyWeightingSkewRootId{
    {.long_flag = "uncertainty-weighting-skew-root",
     .uci_option = "UncertaintyWeightingSkewRoot",
     .help_text =
         "Error output is adjusted to root which skews logistic function."}};

void SearchParams::Populate(OptionsParser* options) {
  BaseSearchParams::Populate(options);
  options->Add<BoolOption>(kUseUncertaintyWeightingId) = true;
  options->Add<FloatOption>(kUncertaintyWeightingMidPointId, 0.0f, 1.0f) =
      0.04f;
  options->Add<FloatOption>(kUncertaintyWeightingMinusExponentId, 0.0f,
                            1000000.0f) = 75.0f;
  options->Add<FloatOption>(kUncertaintyWeightingScaleId, 0.0f, 10.0f) = 0.395;
  options->Add<FloatOption>(kUncertaintyWeightingBaseId, 0.0f, 10.0f) = 0.68f;
  options->Add<FloatOption>(kUncertaintyWeightingSkewRootId, 0.01f, 10000.0f) =
      1.0f;
}

SearchParams::SearchParams(const OptionsDict& options)
    : BaseSearchParams(options),
      kUseUncertaintyWeighting(options.Get<bool>(kUseUncertaintyWeightingId)),
      kUncertaintyWeightingSkewExponent(
          1.0 / options.Get<float>(kUncertaintyWeightingSkewRootId)),
      kUncertaintyWeightingMidPoint(
          std::pow<double>(options.Get<float>(kUncertaintyWeightingMidPointId),
                           kUncertaintyWeightingSkewExponent)),
      kUncertaintyWeightingMinusExponent(
          (0.0 + options.Get<float>(kUncertaintyWeightingMinusExponentId)) *
          options.Get<float>(kUncertaintyWeightingMidPointId) /
          kUncertaintyWeightingMidPoint / kUncertaintyWeightingSkewExponent),
      kUncertaintyWeightingScale(
          options.Get<float>(kUncertaintyWeightingScaleId)),
      kUncertaintyWeightingBase(
          options.Get<float>(kUncertaintyWeightingBaseId)) {}
}  // namespace dag_classic
}  // namespace lczero
