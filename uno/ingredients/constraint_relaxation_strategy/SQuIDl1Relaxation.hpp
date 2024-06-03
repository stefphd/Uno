// Copyright (c) 2018-2024 Charlie Vanaret
// Licensed under the MIT license. See LICENSE file in the project directory for details.

#ifndef UNO_SQUIDL1RELAXATION_H
#define UNO_SQUIDL1RELAXATION_H

#include <memory>
#include "ConstraintRelaxationStrategy.hpp"
#include "ingredients/globalization_strategy/GlobalizationStrategy.hpp"
#include "ingredients/subproblem/Subproblem.hpp"
#include "reformulation/l1RelaxedProblem.hpp"

struct Squidl1RelaxationParameters {
   double beta;
   double theta;
   double kappa_rho;
   double kappa_lambda;
   double epsilon;
   double omega;
   double delta;
};

class SQuIDl1Relaxation : public ConstraintRelaxationStrategy {
public:
   SQuIDl1Relaxation(const Model& model, const Options& options);

   void initialize(Statistics& statistics, Iterate& initial_iterate, const Options& options) override;
   void set_trust_region_radius(double trust_region_radius) override;

   [[nodiscard]] size_t maximum_number_variables() const override;
   [[nodiscard]] size_t maximum_number_constraints() const override;

   // direction computation
   void compute_feasible_direction(Statistics& statistics, Iterate& current_iterate, Direction& direction,
         WarmstartInformation& warmstart_information) override;
   void compute_feasible_direction(Statistics& statistics, Iterate& current_iterate, Direction& direction, const Vector<double>& initial_point,
         WarmstartInformation& warmstart_information) override;
   [[nodiscard]] bool solving_feasibility_problem() const override;
   void switch_to_feasibility_problem(Statistics& statistics, Iterate& current_iterate) override;

   // trial iterate acceptance
   void compute_progress_measures(Iterate& current_iterate, Iterate& trial_iterate, const Direction& direction, double step_length) override;
   [[nodiscard]] bool is_iterate_acceptable(Statistics& statistics, Iterate& current_iterate, Iterate& trial_iterate, const Direction& direction,
         double step_length) override;

   [[nodiscard]] size_t get_hessian_evaluation_count() const override;
   [[nodiscard]] size_t get_number_subproblems_solved() const override;

protected:
   const l1RelaxedProblem feasibility_problem;
   l1RelaxedProblem l1_relaxed_problem;
   std::unique_ptr<Subproblem> subproblem;
   const std::unique_ptr<GlobalizationStrategy> globalization_strategy;
   double penalty_parameter;
   const double tolerance;
   const Squidl1RelaxationParameters parameters;

   void solve_subproblem(Statistics& statistics, const OptimizationProblem& problem, Iterate& current_iterate, const Multipliers& current_multipliers,
         Direction& direction, const WarmstartInformation& warmstart_information);

   void evaluate_progress_measures(Iterate& iterate) const;
   [[nodiscard]] ProgressMeasures compute_predicted_reduction_models(Iterate& current_iterate, const Direction& direction, double step_length);

   void set_dual_residuals_statistics(Statistics& statistics, const Iterate& iterate) const override;
   void check_exact_relaxation(Iterate& iterate) const;


   // temporary functions with the notations of the paper
   [[nodiscard]] double v(const Iterate& current_iterate) const;
   [[nodiscard]] double l(const Direction& direction, const Iterate& current_iterate) const;
   [[nodiscard]] double delta_l(const Direction& direction, const Iterate& current_iterate) const;
   [[nodiscard]] double Ropt(Iterate& current_iterate, double objective_multiplier, const Multipliers& multipliers) const;
   [[nodiscard]] double Rinf(Iterate& current_iterate, const Multipliers& multipliers) const;
   [[nodiscard]] double delta_m(const Direction& direction, const Iterate& current_iterate, double objective_multiplier) const;
   [[nodiscard]] double compute_zeta(const Direction& direction, const Iterate& current_iterate) const;
   [[nodiscard]] double compute_w(const Direction& feasibility_direction, const Direction& optimality_direction, const Iterate& current_iterate);
};

#endif //UNO_SQUIDL1RELAXATION_H
