(define (problem place-gear-at-handover)
  (:domain robot-manipulation)

  (:objects
    home ready handover left-of-gear - location
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
; This PDDL problem file was generated on 2026-02-25 11:16:23.319604
