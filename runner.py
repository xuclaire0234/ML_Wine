import runpy

print("Running baseline.py\n")
runpy.run_path(path_name="baseline.py")
print("-"*50)

print("Running knn.py\n")
runpy.run_path(path_name="knn.py")
print("-"*50)

print("Running svm.py\n")
runpy.run_path(path_name="svm.py")
print("-"*50)

print("Running softmax.py\n")
runpy.run_path(path_name="softmax.py")
print("-"*50)
