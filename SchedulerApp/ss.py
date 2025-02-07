# In some other part of your Django code (views, functions, etc.)
from .views import genetic_algorithm_hyperparameter_validation

# Simulating the request object if necessary (e.g., in a script or from another view)
from django.http import HttpRequest

# If you're calling this function programmatically within your code:
request = HttpRequest()  # Or you could pass an actual request object if it's coming from a web request
response = genetic_algorithm_hyperparameter_validation(request)

# Use the response as needed
print(response)