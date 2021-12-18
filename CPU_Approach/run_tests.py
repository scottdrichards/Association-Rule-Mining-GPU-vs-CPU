import subprocess
import time
num_runs = 5
path = "./main.out"

def run_test(params):
    args = [path]
    args.append(params['freq'])
    args.append(params['num_classes'])
    args.append(params['num_transactions'])
    args.append(params['skew'])
    args.append(params['max_transactions'])
    args.append(params['min_transactions'])
    args.append(params['num_threads'])
    args.append(params['useIndex'])
    for _ in range(num_runs):
        subprocess.call(args)

default_params = {
    'freq': ".01", 
    'num_classes': "128", 
    'num_transactions': "10000", 
    'skew': "10", 
    'max_transactions': "8", 
    'min_transactions': "1", 
    'num_threads': "12",
    'useIndex': "index",
}

tests = {
    "num_classes":["4","8","16","32","64","128"],
    "num_transactions":["1000","10000","100000"],
    "skew":["0.1", "1", "2", "4", "8", "16"],
    "max_transactions":["2","4","8","16","32","64","128"],
    "num_threads":["1", "2", "4", "8", "16", "32", "64"],
    "useIndex":["index","nope"]
}

for name, values in tests.iteritems():
    print("Now Testing: " + name)
    time.sleep(2)
    for value in values:
        params = default_params
        params[name] = value
        print("Now doing: " + value)
        time.sleep(.5)
        run_test(params)