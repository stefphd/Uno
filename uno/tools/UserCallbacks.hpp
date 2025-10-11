// Copyright (c) 2024 Charlie Vanaret
// Licensed under the MIT license. See LICENSE file in the project directory for details.

#ifndef UNO_USERCALLBACKS_H
#define UNO_USERCALLBACKS_H

namespace uno {
   // forward declarations
   class Multipliers;
   template <class ElementType>
   class Vector;

   class UserCallbacks {
   public:
      UserCallbacks() = default;
      virtual ~UserCallbacks() = default;

      // return false to stop optimization
      virtual bool notify_acceptable_iterate(const Vector<double>& primals, const Multipliers& multipliers, double objective_multiplier, double primal_feasibility, double dual_feasibility, double complementarity) = 0;
      virtual bool notify_new_primals(const Vector<double>& primals) = 0;
      virtual bool notify_new_multipliers(const Multipliers& multipliers) = 0;
   };

   class NoUserCallbacks: public UserCallbacks {
   public:
      NoUserCallbacks(): UserCallbacks() { }

      // always return true to continue
      bool notify_acceptable_iterate(const Vector<double>& /*primals*/, const Multipliers& /*multipliers*/, double /*objective_multiplier*/, double /*primal_feasibility*/, double /*dual_feasibility*/, double /*complementarity*/) override { return true; }
      bool notify_new_primals(const Vector<double>& /*primals*/) override { return true; }
      bool notify_new_multipliers(const Multipliers& /*multipliers*/) override { return true; }
   };
} // namespace

#endif //UNO_USERCALLBACKS_H