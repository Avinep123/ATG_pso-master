from django import template

register = template.Library()

@register.filter
def dictKey(d, k):
    '''Returns the given key from a dictionary.'''

    return ', '.join(d[k]) if k in d else ''  # Avoid KeyError if key doesn't exist


@register.simple_tag
def sub(s, d, w, t, break_times, teacher_colors=None):
    '''Returns the subject-teacher for a department, weekday, and time period or indicates a break.'''
    # Check if the current time and weekday is a break
    if (t, w) in break_times:
        return 'Break'  # Return 'Break' if the time is a break

    # If not a break, look for the class
    for c in s:
        if c.department.dept_name.strip().lower() == d.strip().lower() and c.meeting_time.day == w and c.meeting_time.time == t:
            instructor_name = c.instructor.name
            class_info = f'{c.course.course_name} ({instructor_name})'

            # If teacher_colors is provided, apply the color
            if teacher_colors:
                color = teacher_colors.get(instructor_name, '#CCCCCC')  # Default to light gray if no color is set
                return f'<span style="background-color:{color};">{class_info}</span>'
            return class_info
    return 'Free'  # Return 'Free' if no match is found









@register.tag
def active(parser, token):
    args = token.split_contents()
    template_tag = args[0]
    if len(args) < 2:
        raise (template.TemplateSyntaxError, f'{template_tag} tag requires at least one argument')
    return NavSelectedNode(args[1:])

class NavSelectedNode(template.Node):
    def __init__(self, patterns):
        self.patterns = patterns
    def render(self, context):
        path = context['request'].path
        for p in self.patterns:
            pValue = template.Variable(p).resolve(context)
            if path == pValue:
                return 'active'
        return ''