(define (problem place-gear-at-handover)
  (:domain robot-manipulation)

  (:objects
    home ready handover {{ data.locations|join(' ') }} - location
    left right up down forward backward - direction{% if data.objects %}
    {%- for obj in data.objects %}
    {{obj}} - object{% endfor %}
    {%- endif %}
  )
  (:init
    {%- if data.init %}
    {%- for init in data.init %}
    ({{ init.predicate }} {{ init.args|join(' ') }})
    {%- endfor %}
    {%- else %}
    (robot-at-location home)
    (gripper-open)
    {%- endif %}
  )
  (:goal (and{% if data.goals %}
    {%- for goal in data.goals %}
    ({{ goal.predicate }} {{ goal.args|join(' ') }}){% endfor %}
    {%- endif %}
  ))
)