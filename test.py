import xlrd

wb=xlrd.open_workbook("questions and responses for machine learning.xlsx")
sheet=wb.sheet_by_index(0)
cell=sheet.cell_value(0, 0)

print(cell)
print(sheet.row_values(0))
print(sheet.row_values(1))
print(sheet.col_values(0))
