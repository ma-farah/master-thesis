# first data-loading
#%%
import pandas as pd
from pathlib import Path

data_dir = Path(__file__).resolve().parents[1] / "data"
data = data_dir / "LDRT_raw.xlsx"

#%%

# immunological data, columns starts at row 5
df_im = pd.read_excel(
    data,
    sheet_name="IPT ",
    header=4,
    engine="openpyxl"
)

# Clinical data, columns starts at row 2
df_cl = pd.read_excel(
    data,
    sheet_name="Patient data & Pain",
    header=1,
    engine="openpyxl"
)


# %%
from skrub import TableReport

# clinical dataset
TableReport(df_cl, max_plot_columns=138)

#%%

# immunological dataset
TableReport(df_im, max_plot_columns=138)


