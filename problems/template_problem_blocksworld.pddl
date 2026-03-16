(define (problem blocksworld_pb)
  (:domain blocksworld)

  (:objects
    {%- for obj in data.objects %}
    {{obj}}
    {%- endif %}
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