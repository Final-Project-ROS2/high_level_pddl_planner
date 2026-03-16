(define (domain gripper)
 (:requirements :strips)

 (:predicates
  (at-robby ?room)
  (at ?ball ?room)
  (free ?gripper)
  (carry ?ball ?gripper)
 )

 (:action move
  :parameters (?from ?to)
  :precondition (at-robby ?from)
  :effect (and
      (not (at-robby ?from))
      (at-robby ?to))
 )

 (:action pick
  :parameters (?ball ?room ?gripper)
  :precondition (and
      (at ?ball ?room)
      (at-robby ?room)
      (free ?gripper))
  :effect (and
      (not (at ?ball ?room))
      (not (free ?gripper))
      (carry ?ball ?gripper))
 )

 (:action drop
  :parameters (?ball ?room ?gripper)
  :precondition (and
      (carry ?ball ?gripper)
      (at-robby ?room))
  :effect (and
      (at ?ball ?room)
      (free ?gripper)
      (not (carry ?ball ?gripper)))
 )
)