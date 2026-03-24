(define (problem gripper_pb)
  (:domain gripper)
  (:requirements :strips)

  (:objects
    {%- for obj in data.objects %}
    {{obj}}
    {%- endfor %}
  )
  (:init
    {%- for init in data.init %}
    ({{ init.predicate }} {{ init.args|join(' ') }})
    {%- endfor %}
  )
  (:goal (and{% if data.goals %}
    {%- for goal in data.goals %}
    ({{ goal.predicate }} {{ goal.args|join(' ') }}){% endfor %}
    {%- endif %}
  ))
)