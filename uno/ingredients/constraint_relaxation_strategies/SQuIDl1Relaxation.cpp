// Copyright (c) 2018-2024 Charlie Vanaret
// Licensed under the MIT license. See LICENSE file in the project directory for details.

#include <cassert>
#include "SQuIDl1Relaxation.hpp"
#include "ingredients/globalization_strategies/GlobalizationStrategyFactory.hpp"
#include "ingredients/globalization_strategies/l1MeritFunction.hpp"
#include "ingredients/subproblems/SubproblemFactory.hpp"
#include "optimization/Direction.hpp"
#include "optimization/Iterate.hpp"
#include "optimization/WarmstartInformation.hpp"
#include "options/Options.hpp"
#include "symbolic/VectorView.hpp"
#include "tools/UserCallbacks.hpp"
#include "tools/Statistics.hpp"

/*
 * A Sequential Quadratic Optimization Algorithm with Rapid Infeasibility Detection
 * James V. Burke, Frank E. Curtis, and Hao Wang
 * https://epubs.siam.org/doi/abs/10.1137/120880045
 */

namespace uno {
   // v: l1 norm: ||c(x_k)||_1
   double SQuIDl1Relaxation::v(const Iterate& current_iterate) const {
      return current_iterate.progress.infeasibility;
   }

   // l: constraint violation of linearized constraint: ||c(x_k) + \alpha \nabla c(x_k)^T d||_1
   double SQuIDl1Relaxation::l(const Direction& direction, const Iterate& current_iterate) const {
      const auto linearized_constraint =
            current_iterate.evaluations.constraints + current_iterate.evaluations.constraint_jacobian * direction.primals;
      return this->model.constraint_violation(linearized_constraint, Norm::L1);
   }

   // delta_l: predicted (linear model) reduction of constraint violation
   // ||c(x_k)||_1 - ||c(x_k) + \alpha \nabla c(x_k)^T d||_1
   double SQuIDl1Relaxation::delta_l(const Direction& direction, const Iterate& current_iterate) const {
      return this->v(current_iterate) - this->l(direction, current_iterate);
   }

   // Ropt: measure that combines stationarity error and complementarity error for l1 relaxed problem
   double SQuIDl1Relaxation::Ropt(Iterate& current_iterate, double objective_multiplier, const Multipliers& multipliers) const {
      // stationarity error
      this->l1_relaxed_problem.evaluate_lagrangian_gradient(current_iterate.residuals.lagrangian_gradient, current_iterate, multipliers);
      const auto scaled_lagrangian = objective_multiplier * current_iterate.residuals.lagrangian_gradient.objective_contribution +
                                     current_iterate.residuals.lagrangian_gradient.constraints_contribution;
      double error = norm_inf(scaled_lagrangian);

      // complementarity error
      error += this->l1_relaxed_problem.complementarity_error(current_iterate.primals, current_iterate.evaluations.constraints, multipliers, 0., Norm::INF);
      return error;
   }

   // Rinf: measure that combines stationarity error and complementarity error for feasibility problem
   double SQuIDl1Relaxation::Rinf(Iterate& current_iterate, const Multipliers& multipliers) const {
      // stationarity error
      this->feasibility_problem.evaluate_lagrangian_gradient(current_iterate.feasibility_residuals.lagrangian_gradient, current_iterate, multipliers);
      const auto scaled_lagrangian = current_iterate.feasibility_residuals.lagrangian_gradient.constraints_contribution;
      double error = norm_inf(scaled_lagrangian);

      // complementarity error
      error += this->feasibility_problem.complementarity_error(current_iterate.primals, current_iterate.evaluations.constraints, multipliers, 0., Norm::INF);
      return error;
   }

   // m: predicted reduction of the l1 merit function
   double SQuIDl1Relaxation::delta_m(const Direction& direction, const Iterate& current_iterate, double objective_multiplier) const {
      return -objective_multiplier * dot(direction.primals, current_iterate.evaluations.objective_gradient) +
             this->delta_l(direction, current_iterate);
   }

   double SQuIDl1Relaxation::compute_w(const Direction& feasibility_direction, const Direction& optimality_direction,
         const Iterate& current_iterate) {
      double delta_l_dbar = this->delta_l(feasibility_direction, current_iterate);
      double weight = 0.;
      double lower_bound = 0.;
      double upper_bound = 1.;
      bool termination = false;
      while (not termination) {
         DEBUG2 << "Testing the interpolation weight " << weight << '\n';
         const auto trial_direction = view(weight * feasibility_direction.primals + (1 - weight) * optimality_direction.primals, 0,
               this->model.number_variables);
         // update reduction in linearized feasibility model
         const auto trial_linearized_constraints =
               current_iterate.evaluations.constraints + current_iterate.evaluations.constraint_jacobian * trial_direction;
         double delta_l_d = current_iterate.progress.infeasibility - this->model.constraint_violation(trial_linearized_constraints, Norm::L1);
         DEBUG2 << "Trial predicted infeasibility reduction = " << delta_l_d << '\n';

         // sufficient decrease condition
         if (delta_l_d >= this->parameters.beta * delta_l_dbar) {
            // test if bisection has succeeded
            if (weight == 0. || upper_bound - lower_bound <= 1e-8) {
               termination = true;
               DEBUG << "Decrease condition satisfied, terminate with interpolation weight " << weight << '\n';
               DEBUG2 << "Interpolated direction:  ";
               print_vector(DEBUG2, trial_direction);
            }
            else {
               // keep the first half
               upper_bound = weight;
               weight = (lower_bound + upper_bound) / 2.;
               DEBUG2 << "Decrease condition satisfied, decreasing the interpolation weight\n";
            }
         }
         else if (weight == 1.) {
            termination = true;
            DEBUG << "Terminate with interpolation weight " << weight << '\n';
            DEBUG2 << "Interpolated direction:  ";
            print_vector(DEBUG2, trial_direction);
         }
         else {
            // sufficient decrease condition violated: keep the second half
            lower_bound = weight;
            weight = (lower_bound + upper_bound) / 2.;
            DEBUG2 << "Decrease condition violated, increasing the interpolation weight\n";
         }
      }
      return weight;
   }

   // zeta: upper bound on objective multiplier
   double SQuIDl1Relaxation::compute_zeta(const Direction& direction, const Iterate& current_iterate) const {
      const double numerator = (1. - this->parameters.epsilon) * this->delta_l(direction, current_iterate);
      const double denominator = dot(direction.primals, current_iterate.evaluations.objective_gradient) +
                                 this->subproblem->get_lagrangian_hessian().quadratic_product(direction.primals, direction.primals) / 2.;
      return numerator / denominator;
   }

   // SQuID code

   SQuIDl1Relaxation::SQuIDl1Relaxation(const Model& model, const Options& options) :
   // call delegating constructor
         SQuIDl1Relaxation(model,
               // create the l1 feasibility problem (objective multiplier = 0)
               l1RelaxedProblem(model, 0., options.get_double("l1_constraint_violation_coefficient"), 0., nullptr),
               // create the l1 relaxed problem
               l1RelaxedProblem(model, options.get_double("l1_relaxation_initial_parameter"),
                     options.get_double("l1_constraint_violation_coefficient"), 0., nullptr),
               options) {
   }

   // private delegating constructor
   SQuIDl1Relaxation::SQuIDl1Relaxation(const Model& model, l1RelaxedProblem&& feasibility_problem, l1RelaxedProblem&& l1_relaxed_problem,
         const Options& options) :
         ConstraintRelaxationStrategy(model,
               l1_relaxed_problem.number_variables, l1_relaxed_problem.number_constraints,
               l1_relaxed_problem.number_objective_gradient_nonzeros(), l1_relaxed_problem.number_jacobian_nonzeros(),
               l1_relaxed_problem.number_hessian_nonzeros(),
               options),
         feasibility_problem(std::forward<l1RelaxedProblem>(feasibility_problem)),
         l1_relaxed_problem(std::forward<l1RelaxedProblem>(l1_relaxed_problem)),
         penalty_parameter(0.1),
         tolerance(options.get_double("tolerance")),
         parameters({
               1e-2, /* beta */
               0.1, /* theta */
               10., /* kappa_rho */
               10., /* kappa_lambda */
               1e-2, /* epsilon */
               1. - 1e-18, /* omega */
               0.5 /* delta */
         }) {
   }

   void SQuIDl1Relaxation::initialize(Statistics& statistics, Iterate& initial_iterate, const Options& options) {
      // statistics
      this->subproblem->initialize_statistics(statistics, options);
      statistics.add_column("penalty param.", Statistics::double_width, options.get_int("statistics_penalty_parameter_column_order"));
      statistics.set("penalty param.", this->penalty_parameter);

      // initial iterate
      initial_iterate.feasibility_residuals.lagrangian_gradient.resize(this->feasibility_problem.number_variables);
      initial_iterate.feasibility_multipliers.lower_bounds.resize(this->feasibility_problem.number_variables);
      initial_iterate.feasibility_multipliers.upper_bounds.resize(this->feasibility_problem.number_variables);
      this->subproblem->set_elastic_variable_values(this->l1_relaxed_problem, initial_iterate);
      this->subproblem->generate_initial_iterate(this->l1_relaxed_problem, initial_iterate);
      this->evaluate_progress_measures(initial_iterate);
      this->compute_primal_dual_residuals(initial_iterate);
      this->set_statistics(statistics, initial_iterate);
      this->globalization_strategy->initialize(statistics, initial_iterate, options);
   }

   void SQuIDl1Relaxation::compute_feasible_direction(Statistics& statistics, Iterate& current_iterate, Direction& direction,
         WarmstartInformation& warmstart_information) {
      statistics.set("penalty param.", this->penalty_parameter);
      direction.reset();

      current_iterate.evaluate_objective(this->model);
      current_iterate.evaluate_objective_gradient(this->model);

      // solve feasibility problem
      DEBUG << "Solving the l1 feasibility problem\n";
      this->subproblem->initialize_feasibility_problem(this->feasibility_problem, current_iterate);
      Direction feasibility_direction(direction.number_variables, direction.number_constraints);
      this->solve_subproblem(statistics, this->feasibility_problem, current_iterate, current_iterate.feasibility_multipliers, feasibility_direction,
            warmstart_information);
      this->subproblem->exit_feasibility_problem(this->feasibility_problem, current_iterate);
      // assemble multipliers for feasibility problem
      Multipliers feasibility_multipliers(current_iterate.number_variables, current_iterate.number_constraints);
      feasibility_multipliers.constraints = current_iterate.feasibility_multipliers.constraints + feasibility_direction.multipliers.constraints;
      feasibility_multipliers.lower_bounds = current_iterate.feasibility_multipliers.lower_bounds + feasibility_direction.multipliers.lower_bounds;
      feasibility_multipliers.upper_bounds = current_iterate.feasibility_multipliers.upper_bounds + feasibility_direction.multipliers.upper_bounds;
      const double feasibility_dual_error = this->Rinf(current_iterate, feasibility_multipliers);
      DEBUG << "Infeasibility dual error = " << feasibility_dual_error << '\n';

      // update penalty parameter and duals
      // test equation (3.12)
      if (this->tolerance < this->v(current_iterate) &&
          this->delta_l(feasibility_direction, current_iterate) <= this->parameters.theta * this->v(current_iterate)) {
         // update penalty parameter according to (3.13)
         this->penalty_parameter = std::min(this->penalty_parameter, this->parameters.kappa_rho * feasibility_dual_error * feasibility_dual_error);
         DEBUG << "Penalty parameter updated to " << this->penalty_parameter << '\n';

         // update current multipliers according to (3.14)
         const double multipliers_distance = norm_2(
               current_iterate.multipliers.constraints + (-1) * current_iterate.feasibility_multipliers.constraints);
         const double alpha = std::min(1., this->parameters.kappa_lambda * feasibility_dual_error * feasibility_dual_error / multipliers_distance);
         current_iterate.multipliers.constraints = alpha * current_iterate.multipliers.constraints +
                                                   (1. - alpha) * current_iterate.feasibility_multipliers.constraints;
         DEBUG2 << "Updated multipliers: " << current_iterate.multipliers.constraints << '\n';
      }

      // solve l1 relaxed problem
      DEBUG << "\nSolving the regular l1 relaxed problem\n";
      this->l1_relaxed_problem.set_objective_multiplier(this->penalty_parameter);
      //this->subproblem->set_initial_point(feasibility_direction.primals);
      warmstart_information.only_objective_changed();
      Direction optimality_direction(direction.number_variables, direction.number_constraints);
      this->solve_subproblem(statistics, this->l1_relaxed_problem, current_iterate, current_iterate.multipliers, optimality_direction,
            warmstart_information);
      // assemble multipliers for l1 relaxed problem
      Multipliers multipliers(current_iterate.number_variables, current_iterate.number_constraints);
      multipliers.constraints = current_iterate.multipliers.constraints + optimality_direction.multipliers.constraints;
      multipliers.lower_bounds = current_iterate.multipliers.lower_bounds + optimality_direction.multipliers.lower_bounds;
      multipliers.upper_bounds = current_iterate.multipliers.upper_bounds + optimality_direction.multipliers.upper_bounds;
      const double optimality_dual_error = this->Ropt(current_iterate, this->penalty_parameter, multipliers);
      DEBUG << "Optimality dual error = " << optimality_dual_error << '\n';

      // interpolate between two directions
      DEBUG << "\nInterpolating between the two directions:\n";
      DEBUG2 << "Feasibility direction: ";
      print_vector(DEBUG2, view(feasibility_direction.primals, 0, this->model.number_variables));
      DEBUG2 << "Optimality direction:  ";
      print_vector(DEBUG2, view(optimality_direction.primals, 0, this->model.number_variables));
      const double w = this->compute_w(feasibility_direction, optimality_direction, current_iterate);
      const auto interpolated_direction = w * feasibility_direction.primals + (1 - w) * optimality_direction.primals;
      for (size_t variable_index: Range(direction.number_variables)) {
         direction.primals[variable_index] = interpolated_direction[variable_index];
      }

      // update the penalty parameter (3.18)
      const double multipliers_inf_norm = norm_inf(multipliers.constraints, multipliers.lower_bounds, multipliers.upper_bounds);
      if (this->penalty_parameter * multipliers_inf_norm > 1.) {
         this->penalty_parameter = std::min(this->parameters.delta * this->penalty_parameter, (1. - this->parameters.epsilon) / multipliers_inf_norm);
         DEBUG << "Penalty parameter updated to " << this->penalty_parameter << '\n';
      }
      // update the penalty parameter (3.19)
      const double delta_m = this->delta_m(direction, current_iterate, this->penalty_parameter);
      const double delta_l = this->delta_l(direction, current_iterate);
      if (delta_m >= this->parameters.epsilon * delta_l && w >= this->parameters.omega) {
         this->penalty_parameter *= this->parameters.delta;
      }
      else if (delta_m < this->parameters.epsilon * delta_l) {
         const double zeta = this->compute_zeta(direction, current_iterate);
         this->penalty_parameter = std::min(this->parameters.delta * this->penalty_parameter, zeta);
      }
      DEBUG << "Penalty parameter updated to " << this->penalty_parameter << '\n';

      // construct the final direction
      direction.multipliers = optimality_direction.multipliers;
      direction.feasibility_multipliers = feasibility_direction.multipliers;
      direction.norm = norm_inf(view(direction.primals, 0, this->model.number_variables));
   }

   bool SQuIDl1Relaxation::solving_feasibility_problem() const {
      return (this->penalty_parameter == 0.);
   }

   void SQuIDl1Relaxation::switch_to_feasibility_problem(Statistics& /*statistics*/, Iterate& /*current_iterate*/,
         WarmstartInformation& /*warmstart_information*/) {
      throw std::runtime_error("SQuIDl1Relaxation::switch_to_feasibility_problem is not implemented\n");
   }

   void SQuIDl1Relaxation::solve_subproblem(Statistics& statistics, const OptimizationProblem& problem, Iterate& current_iterate,
         const Multipliers& current_multipliers, Direction& direction, const WarmstartInformation& warmstart_information) {
      DEBUG << "Solving the subproblem with penalty parameter " << problem.get_objective_multiplier() << "\n";

      // solve the subproblem
      this->subproblem->solve(statistics, problem, current_iterate, current_multipliers, direction, warmstart_information);
      direction.norm = norm_inf(view(direction.primals, 0, this->model.number_variables));
      DEBUG3 << direction << '\n';
      assert(direction.status == SubproblemStatus::OPTIMAL && "The subproblem was not solved to optimality");
   }

   bool SQuIDl1Relaxation::is_iterate_acceptable(Statistics& statistics, Iterate& current_iterate, Iterate& trial_iterate, const Direction& direction,
         double step_length, WarmstartInformation& /*warmstart_information*/, UserCallbacks& user_callbacks) {
      this->subproblem->postprocess_iterate(this->l1_relaxed_problem, trial_iterate);
      this->compute_progress_measures(current_iterate, trial_iterate);
      trial_iterate.objective_multiplier = this->l1_relaxed_problem.get_objective_multiplier();

      bool accept_iterate = false;
      if (direction.norm == 0.) {
         DEBUG << "Zero step acceptable\n";
         trial_iterate.evaluate_objective(this->model);
         accept_iterate = true;
         statistics.set("status", "0 primal step");
      }
      else {
         // invoke the globalization strategy for acceptance
         const ProgressMeasures predicted_reduction = this->compute_predicted_reduction_models(current_iterate, direction, step_length);
         accept_iterate = this->globalization_strategy->is_iterate_acceptable(statistics, current_iterate.progress, trial_iterate.progress,
               predicted_reduction, this->penalty_parameter);
      }
      if (accept_iterate) {
         this->check_exact_relaxation(trial_iterate);
         user_callbacks.notify_acceptable_iterate(trial_iterate.primals, trial_iterate.multipliers, this->penalty_parameter);
      }
      this->set_progress_statistics(statistics, trial_iterate);
      return accept_iterate;
   }

   void SQuIDl1Relaxation::compute_primal_dual_residuals(Iterate& iterate) {
      ConstraintRelaxationStrategy::compute_primal_dual_residuals(this->l1_relaxed_problem, this->feasibility_problem, iterate);
   }

   void SQuIDl1Relaxation::evaluate_progress_measures(Iterate& iterate) const {
      this->set_infeasibility_measure(iterate);
      this->set_objective_measure(iterate);
      this->subproblem->set_auxiliary_measure(this->model, iterate);
   }

   ProgressMeasures SQuIDl1Relaxation::compute_predicted_reduction_models(Iterate& current_iterate, const Direction& direction, double step_length) {
      return {
            this->compute_predicted_infeasibility_reduction_model(current_iterate, direction.primals, step_length),
            this->compute_predicted_objective_reduction_model(current_iterate, direction.primals, step_length,
                  this->subproblem->get_lagrangian_hessian()),
            this->subproblem->compute_predicted_auxiliary_reduction_model(this->model, current_iterate, direction.primals, step_length)
      };
   }

   size_t SQuIDl1Relaxation::maximum_number_variables() const {
      return this->l1_relaxed_problem.number_variables;
   }

   size_t SQuIDl1Relaxation::maximum_number_constraints() const {
      return this->l1_relaxed_problem.number_constraints;
   }

   // for information purposes, check that l1 is an exact relaxation
   void SQuIDl1Relaxation::check_exact_relaxation(Iterate& iterate) const {
      const double norm_inf_multipliers = norm_inf(iterate.multipliers.constraints);
      if (0. < norm_inf_multipliers && this->penalty_parameter <= 1. / norm_inf_multipliers) {
         DEBUG << "The value of the penalty parameter is consistent with an exact relaxation\n";
      }
   }

   void SQuIDl1Relaxation::set_dual_residuals_statistics(Statistics& statistics, const Iterate& iterate) const {
      statistics.set("complementarity", iterate.residuals.complementarity);
      statistics.set("stationarity", iterate.residuals.stationarity);
   }
} // namespace