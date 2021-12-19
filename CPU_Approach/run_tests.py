import subprocess
import time
num_runs = 3
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

threadCounts = ["1", "2", "4", "8", "16", "32", "64"]


tests = {
    "num_classes":["4","8","16","32","64","128"],
    "skew":["0.1", "1", "2", "4", "8", "16"],
    "max_transactions":["2","4","8"],
    "useIndex":["index","nope"]
}

for name, values in tests.iteritems():
    print("Now Testing: " + name)
    time.sleep(2)
    for value in values:
        for num_threads in threadCounts:
            params = default_params.copy()
            params[name] = value
            params["num_threads"] = num_threads
            print("Now doing: " + value)
            time.sleep(.5)
            run_test(params)