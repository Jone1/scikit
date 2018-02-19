import csv

fileName = 'numbers.txt'
minimal = 4


def find_minimum(input_numbers, number):
    for i, n in enumerate(input_numbers):
        if number - n < 0:
            return input_numbers[i - 1]


def load_numbers(filename):
    with open(filename, 'r') as crimefile:
        reader = csv.reader(crimefile)
        allRows = [row for row in reader]
    numbers = [int(x[0]) for x in allRows]
    return numbers

def save_numbers(results, filename='results.csv'):
    with open(filename, 'w',  newline='') as csvfile:
        fieldnames = ['number', 'iteration', 'result']
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(fieldnames)
        for row in results:
            writer.writerow(row)

def count_result(input_numbers, x):
    temp_x = x
    counter = 0
    while temp_x >= minimal:
        minimum = find_minimum(input_numbers, temp_x)
        temp_x -= minimum
        counter += 1
    return x, counter, temp_x


def _count_result(input_numbers):
    # for x in range(0, 2 ** 16):
    for x in range(0, 100):
        yield count_result(input_numbers, x)

def main():
    input_numbers = load_numbers(fileName)
    results = list(_count_result(input_numbers))
    save_numbers(results)


if __name__ == "__main__":
    main()
