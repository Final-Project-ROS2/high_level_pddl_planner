(define (domain robot-manipulation)
  (:requirements :strips :typing :conditional-effects)
  
  (:types
    location direction object
  )
  
  (:predicates
    (robot-at-location ?loc - location)
    (robot-at-object ?obj - object)
    (object-at-location ?obj - object ?loc - location)
    (robot-have ?obj - object)
    (gripper-open)
    (gripper-close)
  )
  
  (:action move_to_home
    :parameters ()
    :precondition (and)
    :effect (and
      (robot-at-location home)
      (not (robot-at-location ready))
      (not (robot-at-location handover))
      (forall (?obj - object) (not (robot-at-object ?obj)))
    )
  )
  
  (:action move_to_ready
    :parameters ()
    :precondition (and)
    :effect (and
      (robot-at-location ready)
      (not (robot-at-location home))
      (not (robot-at-location handover))
      (forall (?obj - object) (not (robot-at-object ?obj)))
    )
  )

  (:action move_to_handover
    :parameters ()
    :precondition (and)
    :effect (and
      (robot-at-location handover)
      (not (robot-at-location home))
      (not (robot-at-location ready))
      (forall (?obj - object) (not (robot-at-object ?obj)))
    )
  )
  
  (:action open_gripper
    :parameters ()
    :precondition (gripper-closed)
    :effect (and
      (gripper-open)
      (not (gripper-closed))
    )
  )
  
  (:action close_gripper
    :parameters ()
    :precondition (gripper-open)
    :effect (and
      (gripper-closed)
      (not (gripper-open))
    )
  )
  
  (:action move_to_direction
    :parameters (?dir - direction)
    :precondition (and)
    :effect (and
      (not (robot-at-location ready))
      (not (robot-at-location handover))
      (forall (?obj - object) (not (robot-at-object ?obj)))
      (not (robot-at-location home))
      (not (robot-at-location ready))
    )
  )

  (:action move_to_object
    :parameters (?obj - object)
    :precondition (and
      (robot-at-location ready)
    )
    :effect (and
      (robot-at-object ?obj)
      (not (robot-at-location home))
      (not (robot-at-location ready))
      (not (robot-at-location handover))
    )
  )

  (:action pick-object
    :parameters (?obj - object)
    :precondition (and
      (robot-at-location ready)
    )
    :effect (and
      (robot-at-object ?obj)
      (robot-have ?obj)
      (not (robot-at-location ready))
      (not (robot-at-location handover))
      (forall (?obj2 - object) (when (not (= ?obj ?obj2)) (not (robot-at-object ?obj2))))
      (not (robot-at-location home))
      (not (robot-at-location ready))
      (not (robot-at-location handover))
    )
  )

  (:action place-object
    :parameters (?obj - object ?place - location)
    :precondition (and
      (robot-have ?obj)
    )
    :effect (and
      (robot-at-location ?place)
      (not (robot-have ?obj))
      (forall (?obj2 - object) (when (not (= ?obj ?obj2)) (not (robot-at-object ?obj2))))
      (object-at-location ?obj ?place)
    )
  )
)