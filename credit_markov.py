import numpy as np
import scipy.stats as stats
import random
import matplotlib.pyplot as plt

# Given dataset from FitchRatings:
# Fitch Global Corporate Finance One-Year Transition Matrix: 2024 Cohort

# -- Original State Transition Matrix in Percentages as provided by FitchRatings --#


#One-Year Transition Matrix: 2024 Cohort

transition_matrix_percentages_2024 = np.array([
  # States: AAA, AA, A, BBB, BB, B, CCC/C, WD, D
    [95.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 5.00], # AAA
    [ 0.00, 98.62,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 1.38], # AA
    [ 0.00,  1.38, 91.95,  1.26,  0.00,  0.00,  0.00,  0.00, 5.40], # A
    [ 0.00,  0.14,  2.68, 90.74,  1.95,  0.22,  0.00,  0.07, 4.19], # BBB
    [ 0.00,  0.00,  0.13,  3.59, 85.39,  3.85,  0.27,  0.27, 6.51], # BB
    [ 0.00,  0.00,  0.00,  0.00, 11.11, 74.87,  3.08,  1.88, 9.06], # B
    [ 0.00,  0.00,  0.00,  0.00,  0.00,  2.73, 59.09, 30.00, 8.18], # CCC/C
], dtype=np.float64)

#Annual Average 1990-2024
# States: AAA, AA, A, BBB, BB, B, CCC/C, WD, D
transition_matrix_percentages_avg = np.array([
    # AAA   AA     A    BBB   BB     B   CCC to C   D     WD
    [88.75, 5.20, 0.21, 0.00, 0.00, 0.00, 0.00, 0.11, 5.73],  # AAA
    [0.12, 86.70, 8.29, 0.30, 0.02, 0.02, 0.00, 0.05, 4.51],  # AA
    [0.00, 1.43, 89.48, 4.65, 0.31, 0.04, 0.03, 0.05, 4.01],  # A
    [0.00, 0.09, 2.65, 88.32, 3.00, 0.29, 0.08, 0.12, 5.45],  # BBB
    [0.00, 0.02, 0.08, 6.07, 78.79, 5.44, 0.95, 0.58, 8.07],  # BB
    [0.00, 0.00, 0.13, 0.21, 6.82, 75.98, 4.57, 2.02, 10.27], # B
    [0.00, 0.00, 0.00, 0.14, 1.13, 14.59, 48.23, 23.87, 12.04] # CCC to C
], dtype=np.float64)

# However, in order to do matrix multiplication, a square matrix is required. 
# We will add an additional row (default state) to be the absorbing state. 
# Additionally, WD and D will be combined into one absorbing state as WD has no meaning in this context.
# Also, divide by 100 to convert percentage to decimal probabilities.

# Adding a default row for the absorbing state. Both matrixes will use the same row, since there is 100% chance of remaining in default state.
default_row = np.array([[0, 0, 0, 0, 0, 0, 0, 100, 0]], dtype=np.float64)

# For 2024 cohort:
P_8x9_2024 = np.vstack([transition_matrix_percentages_2024, default_row])
P_8x9_2024[:, 7] = P_8x9_2024[:, 7] + P_8x9_2024[:, 8]
P_2024 = np.delete(P_8x9_2024, 8, axis=1)
P_2024 = P_2024 / 100


# For annual average 1990-2024:
P_8x9_avg = np.vstack([transition_matrix_percentages_avg, default_row])
P_8x9_avg[:, 7] = P_8x9_avg[:, 7] + P_8x9_avg[:, 8]
P_avg = np.delete(P_8x9_avg, 8, axis=1)
P_avg = P_avg / 100

P_array = [P_2024, P_avg]


# Valid ratings to be used for user input validation. Althought input of WD is invalid, assume users wont input it as it has no meaning
VALID_STARTING_RATINGS= ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C", "WD/D"]

# Functions:

# Compute n-step transition matrix
def n_step_transition_matrix(P, m):
    return np.linalg.matrix_power(P, m)
  
# Get number of years or number of simulations from user
def get_num_from_user(years=True):
  while True:
    if years:
      num = input("\n Enter the number of years for projection (Ex 1, 2, 5, ...) ")
    else: 
      num = input("\n Enter the number of simulations to run (Ex 100, 2000, 5000, ...) ")
    try:
      num = int(num)
      if num >= 1:
        return num
      else:
        print("\n Please enter a whole number greater than 0.")
        
    except ValueError:
            print("\n Invalid input. Please enter a whole number only.")

# Get year of data to use from user
def get_year_of_data_index():
    print("\n Please enter year of data to use! Options are:\n"
        "CUR - Fitch Global Corporate Finance One-Year Transition Matrix: 2024 Cohort\n"
        "AVG - Annual Average 1990-2024")
    while True:
        year_input = input("Enter `CUR` or 'AVG'").strip().upper()
        if year_input in ["CUR", "AVG"]:
            if year_input == "CUR":
                print("Using Fitch Global Corporate Finance One-Year Transition Matrix: 2024 Cohort")
                return 0
            else:
                print("Using Annual Average 1990-2024")
                return 1

# Return credit rating from user input
def get_rating_from_user(starting=True):
    while True:
      if starting:
        rating = input("\n Enter the starting credit rating (AAA, AA, A, BBB, BB, B, CCC/C, WD/D): ").strip().upper()
      else:
        rating = input("\n Enter the ending credit rating (AAA, AA, A, BBB, BB, B, CCC/C, WD/D): ").strip().upper()
      if rating in VALID_STARTING_RATINGS:
        return rating
            
# Reutrn index of rating in VALID_STARTING_RATINGS
def get_index_of_rating(rating):
    return VALID_STARTING_RATINGS.index(rating)
  
# Return probabilty of transition from starting rating to ending rating in projection years
def obtain_output_value(projection_years, starting_rating, ending_rating, year_of_data):
    P = P_array[year_of_data]
    P_n = n_step_transition_matrix(P, projection_years)
    end_index = get_index_of_rating(ending_rating)
    start_index = get_index_of_rating(starting_rating)
    val = P_n[start_index, end_index]
    return val * 100 
            
# Function calls for user input and output of probability
projection_years = get_num_from_user(years=True)
print(f"{projection_years}- year projection set")
starting_rating = get_rating_from_user(starting=True)
print(f"{starting_rating}- starting rating set")
ending_rating = get_rating_from_user(starting=False)
print(f"{ending_rating}- ending rating set")
year_of_data = get_year_of_data_index()


# Results
probability = obtain_output_value(projection_years, starting_rating, ending_rating, year_of_data)
print("\n--- Results ---")
print(f"Probablity of transitiong from credit rating: {starting_rating} to credit rating: {ending_rating} in {projection_years} years is:\n"
      f"{probability}%\n")












#### Monte Carlo Portion: ####


# Monte Carlo Simulation will be handled using CDF Matrix:
# Since row is the input and columns are pdf of outputs, then the CDF
# is simply summing the probabilities across each row:


# Vars
C_2024 = np.cumsum(P_2024, axis=1)
C_avg = np.cumsum(P_avg, axis=1)
C_array = [C_2024, C_avg]


# Fucntions
# Monte Carlo simulation function:
# Given number of simulations, starting rating, number of years, year of data (to choose which data matrix)
# Returns np array of paths followed by simulations with those paramaters
def monte_carlo_simulation(num_simulations, projection_years, starting_rating, year_of_data):
  C = C_array[year_of_data]
  paths_array = []
  for j in range(num_simulations):
    current_rating_index = get_index_of_rating(starting_rating)
    branch_index_array = []
    for i in range(projection_years):
      branch_index_array.append(current_rating_index)
      U = np.random.uniform(0, 1)
      current_rating_index = np.searchsorted(C[current_rating_index], U)
    branch_index_array.append(current_rating_index)
    paths_array.append(branch_index_array)
  return np.array(paths_array)

# Given paths of simulations and number of simulations, return proportions of ending ratings
def find_proportions(paths, num_simulations):
  outcomes = paths[:, -1]
  proportions = np.array([np.sum(outcomes == i) / num_simulations for i in range(len(VALID_STARTING_RATINGS))])
  return proportions


# Function Calls
num_simulations = get_num_from_user(years=False)
paths = monte_carlo_simulation(num_simulations, projection_years, starting_rating, year_of_data)
proportions = find_proportions(paths, num_simulations)


# Results
print("\n--- Monte Carlo Simulation Results ---")
print(f"After running {num_simulations} simulations for {projection_years} years starting from rating {starting_rating}, the proportions of ending ratings are:")
for i, rating in enumerate(VALID_STARTING_RATINGS):
    print(f"Rating {rating}: {proportions[i]*100:.2f}%")


## Make plot of simulation:
plt.figure(figsize=(10, 6))
for path in paths:
    plt.plot(range(len(path)), path, alpha=0.7)
plt.title("Monte Carlo Simulation of Credit Rating Transitions Over Time")
plt.xlabel("Year After Starting Point")
plt.ylabel("Creidt Rating")
plt.gca().invert_yaxis()
plt.grid(True, linestyle="--", alpha=0.05)
plt.yticks(range(len(VALID_STARTING_RATINGS)), VALID_STARTING_RATINGS)
plt.show()





  