import pandas as pd

def E_open():
    E_df = pd.read_excel('illuminances_std.xlsx', sheet_name='Worksheet')
    return E_df

def R_open():
    R_df = pd.read_excel('babelcolor.xlsx', sheet_name='Worksheet')
    return R_df

def sens_open():
    sensitivities_df = pd.read_excel('canon600d.xlsx', sheet_name='Worksheet').drop(columns='wavelength')
    return sensitivities_df