from django.forms import ModelForm
from .models import *
from django import forms
from django.contrib.auth.forms import AuthenticationForm
from .models import Section, Department, Course

class UserLoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super(UserLoginForm, self).__init__(*args, **kwargs)

    username = forms.CharField(widget=forms.TextInput(
        attrs={
            'class': 'form-control',
            'type': 'text',
            'placeholder': 'UserName',
            'id': 'id_username'
        }))
    password = forms.CharField(widget=forms.PasswordInput(
        attrs={
            'class': 'form-control',
            'type': 'password',
            'placeholder': 'Password',
            'id': 'id_password',
        }))


class RoomForm(ModelForm):
    class Meta:
        model = Room
        labels = {'r_number': 'Room Number'}
        fields = ['r_number', 'seating_capacity']


class InstructorForm(ModelForm):
    
    availability_start = forms.TimeField(
        widget=forms.TimeInput(format='%H:%M', attrs={'type': 'time'}),
        label='Availability Start Time'
    )
    availability_end = forms.TimeField(
        widget=forms.TimeInput(format='%H:%M', attrs={'type': 'time'}),
        label='Availability End Time'
    )
    class Meta:
        model = Instructor
        labels = {'uid': 'Instructor ID', 'name': 'Instructor Name',  'availability_start': 'Availability Start Time',
            'availability_end': 'Availability End Time',}
        fields = ['uid', 'name','availability_start', 'availability_end']
        
    def clean(self):
        cleaned_data = super().clean()
        start_time = cleaned_data.get('availability_start')
        end_time = cleaned_data.get('availability_end')

        # Ensure end time is after start time
        if start_time and end_time and end_time <= start_time:
            raise forms.ValidationError("End time must be after start time.")

        return cleaned_data


class MeetingTimeForm(ModelForm):
    class Meta:
        model = MeetingTime
        fields = ['pid', 'time', 'day']
        widgets = {
            'pid': forms.TextInput(),
            'time': forms.Select(),
            'day': forms.Select(),
        }


class CourseForm(ModelForm):
    class Meta:
        model = Course
        # labels = {'max_numb_students': 'Maximum students'}
        fields = [
            'course_number', 'course_name', 'max_period', 'instructors'
        ]


class DepartmentForm(ModelForm):
    class Meta:
        model = Department
        labels = {'dept_name': 'Department name'}
        fields = ['dept_name', 'courses']


class SectionForm(ModelForm):
    class Meta:
        model = Section
        labels = {'num_class_in_week': 'Total classes in a week'}
        fields = ['section_id', 'department', 'num_class_in_week']
