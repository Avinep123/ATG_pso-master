from django.http.response import HttpResponse
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import *
from collections import defaultdict
import random
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from django.conf import settings
import math,time
import copy

# PSO Hyperparameters
SWARM_SIZE = 25
MAX_ITERATIONS = 300
INITIAL_INERTIA = 0.7
FINAL_INERTIA = 0.7
COGNITIVE_WEIGHT = 1.5
SOCIAL_WEIGHT = 1.5
MAX_VELOCITY = 1.0
MIN_PENALTY = 0.1  # Terminate if penalty is below this threshold

VARS = {'generationNum': 0, 'terminateGens': False}
fitness_values = []

# Sigmoid function for probabilistic updates
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Repair function to enforce hard constraints
def repair(schedule):
    for cls in schedule.getClasses():
        # Ensure room is assigned
        if cls.room is None:
            # Assign a default room if possible
            rooms = data.get_rooms()
            if rooms:
                cls.set_room(random.choice(rooms))
            else:
                raise ValueError("No rooms available to assign.")

        # Proceed with capacity check
        if cls.room.seating_capacity < cls.course.max_numb_students:
            # Find a larger room
            larger_rooms = [r for r in data.get_rooms() if r.seating_capacity >= cls.course.max_numb_students]
            if larger_rooms:
                cls.set_room(random.choice(larger_rooms))

        # Check instructor availability
        instructor = cls.instructor
        meeting_time = cls.meeting_time
        start_time_str, end_time_str = meeting_time.time.split(' - ')
        start_time = datetime.strptime(start_time_str.strip(), '%H:%M').time()
        end_time = datetime.strptime(end_time_str.strip(), '%H:%M').time()

        if start_time < instructor.availability_start or end_time > instructor.availability_end:
            # Find a new meeting time within the instructor's availability
            available_times = [ mt for mt in data.get_meetingTimes()
                                if datetime.strptime(mt.time.split(' - ')[0].strip(), '%H:%M').time() >= instructor.availability_start
                                and datetime.strptime(mt.time.split(' - ')[1].strip(), '%H:%M').time() <= instructor.availability_end]
            if available_times:
                cls.set_meetingTime(random.choice(available_times))

class Data:
    def __init__(self):
        self._rooms = Room.objects.all()
        self._meetingTimes = MeetingTime.objects.all()
        self._instructors = Instructor.objects.all()
        self._courses = Course.objects.all()
        self._depts = Department.objects.all()
        self._sections = Section.objects.all()

    def get_rooms(self):
        return self._rooms

    def get_instructors(self):
        return self._instructors

    def get_courses(self):
        return self._courses

    def get_depts(self):
        return self._depts

    def get_meetingTimes(self):
        return self._meetingTimes

    def get_sections(self):
        return self._sections


class Class:
    def __init__(self, dept, section, course):
        self.department = dept
        self.course = course
        self.instructor = None
        self.meeting_time = None
        self.room = None
        self.section = section

    def get_id(self):
        return self.section_id # see this later

    def get_dept(self):
        return self.department

    def get_course(self):
        return self.course

    def get_instructor(self):
        return self.instructor

    def get_meetingTime(self):
        return self.meeting_time

    def get_room(self):
        return self.room

    def set_instructor(self, instructor):
        self.instructor = instructor

    def set_meetingTime(self, meetingTime):
        self.meeting_time = meetingTime

    def set_room(self, room):
        self.room = room
        
    def __str__(self):
        return f"Class(dept={self.department}, course={self.course}, section={self.section}, instructor={self.instructor}, meeting_time={self.meeting_time}, room={self.room})"



class Schedule:
    def __init__(self):
        self._data = data
        self._classes = []
        self._numberOfConflicts = 0
        self._fitness = -1
        self._isFitnessChanged = True

    def getClasses(self):
        self._isFitnessChanged = True
        return self._classes

    def getNumbOfConflicts(self):
        return self._numberOfConflicts

    def getFitness(self):
        if self._isFitnessChanged:
            self._fitness = self.calculateFitness()
            self._isFitnessChanged = False
        return self._fitness

    def addCourse(self, data, course, courses, dept, section):
        newClass = Class(dept, section.section_id, course)

        newClass.set_meetingTime(
            data.get_meetingTimes()[random.randrange(0, len(data.get_meetingTimes()))])

    # Check if rooms are available
        rooms = data.get_rooms()
        if not rooms:
            raise ValueError("No rooms available in the database.")
        newClass.set_room(rooms[random.randrange(0, len(rooms))])

        crs_inst = course.instructors.all()
        if not crs_inst:
            raise ValueError(f"No instructors available for {course}.")
        newClass.set_instructor(
            crs_inst[random.randrange(0, len(crs_inst))])

        self._classes.append(newClass)


    def get_schedule_summary(self):
        """Returns compact representation of schedule decisions"""
        return [
            (cls.meeting_time.pid, cls.instructor.uid, cls.room.r_number)
            for cls in self._classes
        ]

    def initialize(self):
        sections = Section.objects.all()
        for section in sections:
            dept = section.department
            n = section.num_class_in_week
            # Ensure we don't exceed available meeting times
            available_meeting_times = len(data.get_meetingTimes())
            # print(f"Available meeting times: {available_meeting_times}")

            if n > available_meeting_times:
                print(f"Reducing n from {n} to available meeting times {available_meeting_times}.")
                n = available_meeting_times  # Ensure we don't exceed available meeting times

     

            courses = dept.courses.all()

        # Ensure courses exist
            if not courses:
                print(f"No courses found for department {dept}. Skipping section {section}.")
                continue

        # Calculate how many classes to add
            classes_to_add = n // len(courses)
            for course in courses:
                for i in range(classes_to_add):
                    self.addCourse(data, course, courses, dept, section)

            additional_classes = n % len(courses)
            for course in courses.order_by('?')[:additional_classes]:
                self.addCourse(data, course, courses, dept, section)

        print(f"Total classes initialized: {len(self._classes)}")
        return self
    
    def parse_time(self, time_str):
        """Convert a time string (e.g., '11:30') to a time object."""
        return datetime.strptime(time_str.strip(), '%H:%M').time()


    def calculateFitness(self):
        self._hard_constraint_violations = {
            'same_course_same_section': 0,
            'instructor_conflict': 0,
            'duplicate_time_section': 0,
            'instructor_availability': 0,
            'total_classes_mismatch': 0,
            'course_frequency': 0
        }
        self._soft_constraint_violations = {
            'no_consecutive_classes': 0,
            'noon_classes': 0,
            'break_time_conflict': 0,
            'balanced_days': 0,
        }

        hard_weights = {
            'same_course_same_section': 3,
            'instructor_conflict': 3,
            'duplicate_time_section': 3,
            'instructor_availability': 3,
            'total_classes_mismatch': 3,
            'course_frequency': 3, 
        }

        soft_weights = {
            'no_consecutive_classes': 0.5,
            'noon_classes': 0.5,
            'break_time_conflict': 0.3,
            'balanced_days': 0.3,
        }

    # Check constraints
        classes = self.getClasses()
        self.check_total_classes(classes, hard_weights)
        for i in range(len(classes)):
            self.check_course_conflicts(classes, i, hard_weights)
            self.check_instructor_conflict(classes, i, hard_weights)
            self.check_duplicate_time(classes, i, hard_weights)
            self.check_instructor_availability(classes, i, hard_weights)
        self.check_course_frequency(classes, hard_weights)
        for i in range(len(classes)):
            self.check_consecutive_classes(classes, i, soft_weights)
            self.check_noon_classes(classes, i, soft_weights)
            self.check_break_time_conflict(classes, i, soft_weights)
        self.check_balanced_days(classes, soft_weights)


        hard_penalty = sum(hard_weights[key] * self._hard_constraint_violations[key] for key in hard_weights)
        soft_penalty = sum(soft_weights[key] * self._soft_constraint_violations[key] for key in soft_weights)
        hard_penalty /= max(1, len(hard_weights))
        soft_penalty /= max(1, len(soft_weights))
        total_penalty = soft_penalty + hard_penalty
        alpha = 0.05
        fitness = math.exp(-alpha * total_penalty)
        self._fitness = fitness
        return self._fitness
    
    def check_course_frequency(self, classes, hard_weights):
    # Create a dictionary to track the number of occurrences for each course
        course_count = {}

        for cls in classes:
            course = cls.course
            if course not in course_count:
                course_count[course] = 0
            course_count[course] += 1

        # For each course, check if it appears the required number of times per week
        for course, count in course_count.items():
            required_count = course.max_period  # assuming each course has this attribute
            if count != required_count:
                #print(f"Violation: {course} should appear {required_count} times but appears {count} times.")
                self._hard_constraint_violations['course_frequency'] += 1
    
    
    def check_total_classes(self, classes, weights):
        # Initialize a dictionary to track the number of classes per section
        section_classes = {}

        # print("Checking total classes per section...")  # Debug print

        for cls in classes:
            section = cls.section  # Assuming each class has a 'section' attribute
            
            # If section is a string (like section_id), retrieve the actual Section object
            if isinstance(section, str):
                section = Section.objects.get(section_id=section)
            
            if section not in section_classes:
                section_classes[section] = 0
            section_classes[section] += 1

        # print(f"Total classes per section: {section_classes}")  # Debug print

        # Check if the number of classes in each section matches the expected number
        for section, num_classes in section_classes.items():
            # Retrieve the section's allowed number of classes (e.g., from the Section model)
            allowed_classes = section.num_class_in_week
            # print(f"Section: {section}, Total Classes: {num_classes}, Allowed Classes: {allowed_classes}")  # Debug print

            # Violation if total classes do not match the expected number
            if num_classes != allowed_classes:
                # print(f"Violation: Section {section} has a mismatch in total classes.")  # Debug print
                self._hard_constraint_violations['total_classes_mismatch'] += 1


    


    def check_course_conflicts(self, classes, i, weights):
        for j in range(i + 1, len(classes)):
            day_i = str(classes[i].meeting_time).split()[0]
            day_j = str(classes[j].meeting_time).split()[0]
            if (classes[i].course.course_name == classes[j].course.course_name and 
                day_i == day_j and classes[i].section == classes[j].section):
                self._hard_constraint_violations['same_course_same_section'] += 1

    def check_instructor_conflict(self, classes, i, weights):
        for j in range(i + 1, len(classes)):
            if (classes[i].section != classes[j].section and 
                classes[i].meeting_time == classes[j].meeting_time and 
                classes[i].instructor == classes[j].instructor):
                self._hard_constraint_violations['instructor_conflict'] += 1

    def check_duplicate_time(self, classes, i, weights):
        for j in range(i + 1, len(classes)):
            if (classes[i].section == classes[j].section and 
                classes[i].meeting_time == classes[j].meeting_time):
                self._hard_constraint_violations['duplicate_time_section'] += 1

    def check_instructor_availability(self, classes, i, weights):
        instructor = classes[i].instructor
        availability_start = instructor.availability_start
        availability_end = instructor.availability_end
        meeting_time_str = classes[i].meeting_time.time
        start_time_str, end_time_str = meeting_time_str.split(' - ')
        start_time = self.parse_time(start_time_str)
        end_time = self.parse_time(end_time_str)

        if start_time < availability_start or end_time > availability_end:
            self._hard_constraint_violations['instructor_availability'] += 1

    def check_consecutive_classes(self, classes, i, weights):
        for j in range(i + 1, len(classes)):
            if classes[i].instructor == classes[j].instructor:
                time_i_end = self.parse_time(classes[i].meeting_time.time.split(' - ')[1])
                time_j_start = self.parse_time(classes[j].meeting_time.time.split(' - ')[0])
                if time_i_end == time_j_start:
                    self._soft_constraint_violations['no_consecutive_classes'] += 1

    def check_noon_classes(self, classes, i, weights):
        noon_start = self.parse_time('10:00')
        noon_end = self.parse_time('15:00')
        start_time_str, _ = classes[i].meeting_time.time.split(' - ')
        start_time = self.parse_time(start_time_str)

        if noon_start <= start_time <= noon_end:
            self._soft_constraint_violations['noon_classes'] += 1

    def check_break_time_conflict(self, classes, i, weights):
        break_start = self.parse_time('10:00')
        break_end = self.parse_time('10:50')
        start_time_str, _ = classes[i].meeting_time.time.split(' - ')
        start_time = self.parse_time(start_time_str)
        end_time_str, _ = classes[i].meeting_time.time.split(' - ')
        end_time = self.parse_time(end_time_str)

        if start_time < break_end and end_time > break_start:
            self._soft_constraint_violations['break_time_conflict'] += 1

    def check_balanced_days(self, classes, weights):
        day_class_count = {}
        for cls in classes:
            day = str(cls.meeting_time).split()[0]
            if day not in day_class_count:
                day_class_count[day] = 0
            day_class_count[day] += 1
        max_day = max(day_class_count.values())
        min_day = min(day_class_count.values())
        if max_day - min_day > 1 :
            self._soft_constraint_violations['balanced_days'] += 1


    

# Particle class representing a candidate solution
class Particle:
    def __init__(self, classes_data):
        self.position = []
        self.velocity = []
        self.pbest_position = []
        self.pbest_fitness = -1
        self.classes_data = classes_data

        # Initialize position and velocity
        for data in self.classes_data:
            mt_pos = random.uniform(0, len(data['mt_options']))
            instr_pos = random.uniform(0, len(data['instructor_options']))
            room_pos = random.uniform(0, len(data['room_options']))
            self.position.extend([mt_pos, instr_pos, room_pos])

            mt_vel = random.uniform(-1, 1)
            instr_vel = random.uniform(-1, 1)
            room_vel = random.uniform(-1, 1)
            self.velocity.extend([mt_vel, instr_vel, room_vel])

        self.pbest_position = list(self.position)

    def update_schedule(self, schedule):
        idx = 0
        for i, cls in enumerate(schedule.getClasses()):
            data = self.classes_data[i]
        
        # Ensure options are not empty
            if len(data['mt_options']) == 0 or len(data['instructor_options']) == 0 or len(data['room_options']) == 0:
                print(f"Warning: No options available for class {i}. Skipping update.")
                continue  # Skip this class if options are empty

        # Update meeting time
            mt_idx = int(self.position[idx]) % len(data['mt_options'])
            cls.set_meetingTime(data['mt_options'][mt_idx])
        
        # Update instructor
            instr_idx = int(self.position[idx + 1]) % len(data['instructor_options'])
            cls.set_instructor(data['instructor_options'][instr_idx])
        
        # Update room
            room_idx = int(self.position[idx + 2]) % len(data['room_options'])
            cls.set_room(data['room_options'][room_idx])
        
            idx += 3

class Swarm:
    def __init__(self, num_particles, classes_data):
        self.particles = [Particle(classes_data) for _ in range(num_particles)]
        self.gbest_position = None
        self.gbest_fitness = -1

    def update_gbest(self):
        for p in self.particles:
            if p.pbest_fitness > self.gbest_fitness:
                self.gbest_position = list(p.pbest_position)
                self.gbest_fitness = p.pbest_fitness


def context_manager(schedule):
    classes = schedule.getClasses()
    context = []
    for i in range(len(classes)):
        clas = {}
        clas['section'] = classes[i].section.section_id
        clas['dept'] = classes[i].department.dept_name
        clas['course'] = f'{classes[i].course.course_name} ({classes[i].course.course_number} {classes[i].course.max_numb_students})'
        clas['room'] = f'{classes[i].room.r_number} ({classes[i].room.seating_capacity})'
        clas['instructor'] = f'{classes[i].instructor.name} ({classes[i].instructor.uid})'
        clas['meeting_time'] = [
            classes[i].meeting_time.pid,
            classes[i].meeting_time.day,
            classes[i].meeting_time.time
        ]
        context.append(clas)
    return context


def apiGenNum(request):
    return JsonResponse({'genNum': VARS['generationNum']})

def apiterminateGens(request):
    VARS['terminateGens'] = True
    return redirect('home')




from random import choice

def get_random_color():
    # Generate light colors by ensuring RGB values are higher than 200
    r = random.randint(200, 255)
    g = random.randint(200, 255)
    b = random.randint(200, 255)
    
    # Return the color in the hex format
    return f"#{r:02x}{g:02x}{b:02x}"






@login_required
def timetable(request):
    global data
    data = Data()
    initial_schedule = Schedule().initialize()
    classes = initial_schedule.getClasses()

    # Precompute possible options for each class's attributes
    classes_data = []
    for cls in classes:
        mt_options = list(data.get_meetingTimes())
        instructor_options = list(cls.course.instructors.all())
        room_options = list(data.get_rooms())
        classes_data.append({
            'mt_options': mt_options,
            'instructor_options': instructor_options,
            'room_options': room_options,
        })

    swarm = Swarm(SWARM_SIZE, classes_data)
    VARS['generationNum'] = 0
    VARS['terminateGens'] = False

    # Initialize global best
    best_fitness = -1
    for i, p in enumerate(swarm.particles):
        schedule = Schedule()
        schedule._classes = copy.deepcopy(initial_schedule.getClasses())
        print(f"Initializing Particle {i} schedule ID: {id(schedule)}")  # Debug log
        p.update_schedule(schedule)
        repair(schedule)
        fitness = schedule.getFitness()
        if fitness > best_fitness:
            best_fitness = fitness
            swarm.gbest_position = list(p.position)
            swarm.gbest_fitness = best_fitness

    fitness_values = []
    average_fitness = []
    diversity = []
    velocity_magnitudes_per_generation = []
    velocity_diversities = []

    start_time = time.time()


    while swarm.gbest_fitness < 1.0 and VARS['generationNum'] < MAX_ITERATIONS and not VARS['terminateGens']:
        velocity_magnitudes = []
        for i, p in enumerate(swarm.particles):  # Ensure the loop variable is correctly defined
            schedule = Schedule()
            schedule._classes = copy.deepcopy(initial_schedule.getClasses())
            print(f"Updating Particle {i} schedule ID: {id(schedule)}")  # Debug log
            p.update_schedule(schedule)
            repair(schedule)
            fitness = schedule.getFitness()

            # Update personal best
            if fitness > p.pbest_fitness:
                p.pbest_position = list(p.position)
                p.pbest_fitness = fitness

            # Update velocity and position
            for d in range(len(p.velocity)):
                r1 = random.random()
                r2 = random.random()
                INERTIA_WEIGHT = INITIAL_INERTIA - ((INITIAL_INERTIA - FINAL_INERTIA) * (VARS['generationNum'] / MAX_ITERATIONS))
                inertia = INERTIA_WEIGHT * p.velocity[d]
                cognitive = COGNITIVE_WEIGHT * r1 * (p.pbest_position[d] - p.position[d])
                social = SOCIAL_WEIGHT * r2 * (swarm.gbest_position[d] - p.position[d])
                p.velocity[d] = inertia + cognitive + social
                p.velocity[d] = np.clip(p.velocity[d], -MAX_VELOCITY, MAX_VELOCITY)
                p.position[d] += p.velocity[d]

            velocity_magnitude = np.linalg.norm(p.velocity)
            velocity_magnitudes.append(velocity_magnitude)

        

        average_velocity_magnitude = np.mean(velocity_magnitudes)
        velocity_magnitudes_per_generation.append(average_velocity_magnitude)

        velocity_diversity = np.std(velocity_magnitudes)
        velocity_diversities.append(velocity_diversity)

        swarm.update_gbest()
        fitness_values.append(swarm.gbest_fitness)
        current_fitnesses = [particle.pbest_fitness for particle in swarm.particles]
        average_fitness.append(sum(current_fitnesses) / len(current_fitnesses))
        diversity.append(np.std(current_fitnesses))
        VARS['generationNum'] += 1

    """ generate_velocity_plots( velocity_magnitudes_per_generation,velocity_diversities) """

    end_time = time.time()
    execution_time = end_time - start_time

    # Print final metrics
    print(f"Final Best Fitness: {swarm.gbest_fitness}")
    print(f"Final Average Fitness: {average_fitness[-1]}")
    print(f"Execution Time: {execution_time:.2f} seconds")

    # Generate plots and render timetable template
    generate_combined_plots(fitness_values, average_fitness, diversity,velocity_magnitudes_per_generation, SWARM_SIZE, INERTIA_WEIGHT)
    best_schedule = Schedule()
    best_schedule._classes = copy.deepcopy(initial_schedule.getClasses())
    best_particle = Particle(classes_data)
    best_particle.position = swarm.gbest_position
    best_particle.update_schedule(best_schedule)
    repair(best_schedule)

    # Render timetable template
    break_time_slot = '10:00 - 10:50'
    week_days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
    teacher_colors = {}
    for cls in best_schedule.getClasses():
        instructor_name = cls.get_instructor().name
        if instructor_name not in teacher_colors:
            teacher_colors[instructor_name] = get_random_color()

    return render(request, 'timetable.html', {
        'schedule': best_schedule.getClasses(),
        'sections': data.get_sections(),
        'times': data.get_meetingTimes(),
        'timeSlots': TIME_SLOTS,
        'weekDays': DAYS_OF_WEEK,
        'break_times': [(break_time_slot, day) for day in week_days],
        'teacher_colors': teacher_colors,
    })

def generate_combined_plots(fitness_values, average_fitness, diversity,velocity_magnitudes_per_generation, swarm_size, inertia):
    plt.figure(figsize=(15, 5))
    
    # Best Fitness Plot
    plt.subplot(1, 3, 1)
    plt.plot(fitness_values, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best Fitness per Generation')
    plt.legend()

    # Average Fitness Plot
    plt.subplot(1, 3, 2)
    plt.plot(average_fitness, label='Average Fitness', color='orange')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Average Fitness per Generation')
    plt.legend()

    # Diversity Plot
    """ plt.subplot(1, 3, 3)
    plt.plot(diversity, label='Unique Fitness Count', color='green')
    plt.xlabel('Generation')
    plt.ylabel('Unique Fitness Values')
    plt.title('Population Diversity')
    plt.legend() """

    plt.subplot(1, 3, 3)
    plt.plot(velocity_magnitudes_per_generation, label='Average Velocity Magnitude', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Velocity Magnitude')
    plt.title('Average Velocity Magnitude per Generation')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(settings.MEDIA_ROOT, 'pso_plots.png')
    plt.savefig(plot_path)
    plt.close()


""" def generate_velocity_plots(velocity_magnitudes_per_generation, velocity_diversities):
    plt.figure(figsize=(12, 6))

    # Plot average velocity magnitude
    plt.subplot(1, 2, 1)
    plt.plot(velocity_magnitudes_per_generation, label='Average Velocity Magnitude', color='blue')
    plt.xlabel('Generation')
    plt.ylabel('Velocity Magnitude')
    plt.title('Average Velocity Magnitude per Generation')
    plt.legend()

    # Plot velocity diversity
    plt.subplot(1, 2, 2)
    plt.plot(velocity_diversities, label='Velocity Diversity', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Velocity Diversity (Std Dev)')
    plt.title('Velocity Diversity per Generation')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(settings.MEDIA_ROOT, 'velocity_plots.png')
    plt.savefig(plot_path)
    plt.close()
 """
'''
Page Views
'''

def home(request):
    return render(request, 'index.html', {})


@login_required
def instructorAdd(request):
    form = InstructorForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('instructorAdd')
    context = {'form': form}
    return render(request, 'instructorAdd.html', context)


@login_required
def instructorEdit(request):
    context = {'instructors': Instructor.objects.all()}
    return render(request, 'instructorEdit.html', context)


@login_required
def instructorDelete(request, pk):
    inst = Instructor.objects.filter(pk=pk)
    if request.method == 'POST':
        inst.delete()
        return redirect('instructorEdit')


@login_required
def roomAdd(request):
    form = RoomForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('roomAdd')
    context = {'form': form}
    return render(request, 'roomAdd.html', context)


@login_required
def roomEdit(request):
    context = {'rooms': Room.objects.all()}
    return render(request, 'roomEdit.html', context)


@login_required
def roomDelete(request, pk):
    rm = Room.objects.filter(pk=pk)
    if request.method == 'POST':
        rm.delete()
        return redirect('roomEdit')


@login_required
def meetingTimeAdd(request):
    form = MeetingTimeForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('meetingTimeAdd')
        else:
            print('Invalid')
    context = {'form': form}
    return render(request, 'meetingTimeAdd.html', context)


@login_required
def meetingTimeEdit(request):
    context = {'meeting_times': MeetingTime.objects.all()}
    return render(request, 'meetingTimeEdit.html', context)


@login_required
def meetingTimeDelete(request, pk):
    mt = MeetingTime.objects.filter(pk=pk)
    if request.method == 'POST':
        mt.delete()
        return redirect('meetingTimeEdit')


@login_required
def courseAdd(request):
    form = CourseForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('courseAdd')
        else:
            print('Invalid')
    context = {'form': form}
    return render(request, 'courseAdd.html', context)


@login_required
def courseEdit(request):
    instructor = defaultdict(list)
    for course in Course.instructors.through.objects.all():
        course_number = course.course_id
        instructor_name = Instructor.objects.filter(
            id=course.instructor_id).values('name')[0]['name']
        instructor[course_number].append(instructor_name)

    context = {'courses': Course.objects.all(), 'instructor': instructor}
    return render(request, 'courseEdit.html', context)


@login_required
def courseDelete(request, pk):
    crs = Course.objects.filter(pk=pk)
    if request.method == 'POST':
        crs.delete()
        return redirect('courseEdit')


@login_required
def departmentAdd(request):
    form = DepartmentForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('departmentAdd')
    context = {'form': form}
    return render(request, 'departmentAdd.html', context)


@login_required
def departmentEdit(request):
    course = defaultdict(list)
    for dept in Department.courses.through.objects.all():
        dept_name = Department.objects.filter(
            id=dept.department_id).values('dept_name')[0]['dept_name']
        course_name = Course.objects.filter(
            course_number=dept.course_id).values(
                'course_name')[0]['course_name']
        course[dept_name].append(course_name)

    context = {'departments': Department.objects.all(), 'course': course}
    return render(request, 'departmentEdit.html', context)


@login_required
def departmentDelete(request, pk):
    dept = Department.objects.filter(pk=pk)
    if request.method == 'POST':
        dept.delete()
        return redirect('departmentEdit')


@login_required
def sectionAdd(request):
    form = SectionForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('sectionAdd')
    context = {'form': form}
    return render(request, 'sectionAdd.html', context)


@login_required
def sectionEdit(request):
    context = {'sections': Section.objects.all()}
    return render(request, 'sectionEdit.html', context)


@login_required
def sectionDelete(request, pk):
    sec = Section.objects.filter(pk=pk)
    if request.method == 'POST':
        sec.delete()
        return redirect('sectionEdit')




'''
Error pages
'''

def error_404(request, exception):
    return render(request,'errors/404.html', {})

def error_500(request, *args, **argv):
    return render(request,'errors/500.html', {})