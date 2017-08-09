import xlrd
import csv

def csv_from_xlsx():

    wb = xlrd.open_workbook('../Data/Rating_Collection/Attractiveness_label.xlsx')
    sh = wb.sheet_by_name('Sheet1')
    csv_file = open('attractiveness_rating.csv', 'wb')
    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    for rownum in xrange(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    csv_file.close()

if __name__ == "__main__":
    csv_from_xlsx()
