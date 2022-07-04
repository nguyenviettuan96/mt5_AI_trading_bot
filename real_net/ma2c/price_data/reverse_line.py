import csv

with open("EURUSD_threemonths.csv") as fr, open("reverse_EURUSD_threemonths.csv", "w", newline="") as fw:
    cr = csv.reader(fr, delimiter=",")
    cw = csv.writer(fw, delimiter=",")
    cw.writerow(next(cr))  # write title as-is
    cw.writerows(reversed(list(cr)))
