(define (problem place-gear-at-handover)
  (:domain robot-manipulation)

  (:objects
    home ready handover - location
    left right up down forward backward - direction{% if data.objects %}
    {%- for obj in data.objects %}
    {{obj}} - object{% endfor %}
    {%- endif %}
  )
  (:init
    (robot-at-location home)
    (gripper-open)
  )
  (:goal (and{% if data.goals %}
    {%- for goal in data.goals %}
    ({{ goal.predicate }} {{ goal.args|join(' ') }}){% endfor %}
    {%- endif %}
  ))
)