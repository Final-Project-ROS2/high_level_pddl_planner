(define (problem place-gear-at-handover)
  (:domain robot-manipulation)

  (:objects
    home ready handover - location
    left right up down forward backward - direction
    gear - object
    bolt - object
  )
  (:init
    (robot-at-location home)
    (gripper-open)
  )
  (:goal (and
    (gripper-closed )
  ))
)
; This PDDL problem file was generated on 2026-02-25 08:27:30.905881
