# Schlomann_Final02

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

df = pd.read_csv("C:\\PATH\\TO\\cost_of_living_us.csv")

# 1)
df = df.dropna()
# Gemini helped
df["adults"] = (df["family_member_count"].str[0]).astype(int)
df["children"] = (df["family_member_count"].str[2]).astype(int)
df["residents"] = (df["adults"]+df["children"])
df_quant = df.drop(columns = ["case_id", "isMetro", "state", "areaname", "county", "family_member_count"])

df_sample = df.sample(frac=0.1, random_state=1)
df_sample.to_csv("C:\\PATH\\TO\\cost_of_living_us_sample.csv")

# 2)
fig2 = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df.case_id, df.state, df.isMetro, df.areaname, df.county, df.family_member_count, df.housing_cost, df.food_cost, 
                       df.transportation_cost, df.healthcare_cost, df.other_necessities_cost, df.childcare_cost, df.taxes, df.total_cost,
                       df.median_family_income, df.adults, df.children, df.residents], 
               fill_color='lavender',
               align='left'))
])
fig2.write_html("C:\\PATH\\TO\\Final_Summary.html")

# 3)
fig31 = px.box(df, x="state", y="transportation_cost", title="Transportation Cost by State")
fig31.write_html("C:\\PATH\\TO\\Final_Boxplot31.html")

fig32 = px.box(df, x="state", y="healthcare_cost", title="Healthcare Cost by State")
fig32.write_html("C:\\PATH\\TO\\Final_Boxplot32.html")

fig33 = px.box(df, x="state", y="childcare_cost", title="Childcare Cost by State")
fig33.write_html("C:\\PATH\\TO\\Final_Boxplot33.html")

fig34 = px.box(df, x="state", y="taxes",  title="Taxes by State")
fig34.write_html("C:\\PATH\\TO\\Final_Boxplot34.html")

fig35 = px.box(df, x="state", y="median_family_income", title="Median Family Income by State")
fig35.write_html("C:\\PATH\\TO\\Final_Boxplot35.html")

fig36 = px.box(df, x="state", y="residents", title="Total Residents by State")
fig36.write_html("C:\\PATH\\TO\\Final_Boxplot36.html")

fig36a = px.box(df, x="state", y="adults", title="Adults by State")
fig36a.write_html("C:\\PATH\\TO\\Final_Boxplot36a.html")

fig36b = px.box(df, x="state", y="children", title="Children by State")
fig36b.write_html("C:\\PATH\\TO\\Final_Boxplot36b.html")

# 4)
fig4 = px.pie(df, values='taxes', names="state", title="Taxes by State")
fig4.update_layout(margin=dict(t=0, b=0, l=0, r=0))
fig4.write_html("C:\\PATH\\TO\\Final_Pie4.html")

# 5)
state_geo_url = "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json"

state_geo = requests.get(state_geo_url).json()

m = folium.Map(location=[48, -102], zoom_start=4)

folium.Choropleth(
    geo_data=state_geo,
    name="Transportation Cost by State",
    data=df,
    columns=['state', 'transportation_cost'],
    key_on="feature.id",
    fill_color="YlGn",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Transportation Cost",
).add_to(m)

folium.LayerControl().add_to(m)

m.save("C:\\PATH\\TO\\Final_Choropleth5.html")

# 6)
fig61 = px.scatter(df, x="total_cost", y="transportation_cost",
	         color="state", title="Total Cost vs. Transportation Cost by State")
fig61.write_html("C:\\PATH\\TO\\Final_Scatter61.html")

fig62 = px.scatter(df, x="childcare_cost", y="taxes",
	         color="state", title="Childcare Cost vs Taxes by State")
fig62.write_html("C:\\PATH\\TO\\Final_Scatter62.html")

fig63 = px.scatter(df, x="median_family_income", y="taxes",
	         color="state", title="Median Family Income vs. Taxes by State")
fig63.write_html("C:\\PATH\\TO\\Final_Scatter63.html")

fig64 = px.scatter(df, x="housing_cost", y="taxes",
	         color="state", title="Housing Cost vs. Taxes by State")
fig64.write_html("C:\\PATH\\TO\\Final_Scatter64.html")


# 7)
df_childcare = df.loc[:, ['childcare_cost', 'county']]
childcare_county = df_childcare.groupby(['county']).mean()
childcare_county.sort_values(by='childcare_cost', ascending=False).head(12)

# 8)
df_taxes = df.loc[:, ['taxes', 'county']]
taxes_county = df_taxes.groupby(['county']).mean()
taxes_county.sort_values(by='taxes', ascending=False).head(12)

# 9)
correlation = df_quant.corr()
sns.heatmap(df_quant.corr(), annot=True, fmt=".1f", cmap="crest")

# 10)
sns.regplot(x="taxes", y="total_cost", data=df)
plt.title("Taxes vs. Total Cost")
plt.show()

# 11)
fig11 = px.box(df, x='state', y='food_cost', title="Food Cost by State", hover_data='case_id')
fig11.write_html("C:\\PATH\\TO\\Final_Boxplot11.html")

# 12)
fig12 = px.scatter(df, x="food_cost", y="taxes",
	         size="childcare_cost", color="isMetro", title="Food Cost vs. Taxes sized by Childcare Cost")
fig12.write_html("C:\\PATH\\TO\\Final_BubbleChart12.html")

# 13)
fig13 = px.scatter(df, x="housing_cost", y="taxes",
	         size="transportation_cost", color="isMetro", title="Housing Cost vs. Taxes sized by Transportation Cost")
fig13.write_html("C:\\PATH\\TO\\Final_BubbleChart13.html")

# 14)
fig14 = px.scatter_3d(df, x='transportation_cost', y='healthcare_cost',
                      z='childcare_cost', title="Transportation Cost vs. Healthcare Cost vs. Childcare Cost")
fig14.write_html("C:\\PATH\\TO\\Final_Scatter3d14.html")

# 15)
fig15 = px.scatter_3d(df, x='childcare_cost', y='taxes',
                      z='median_family_income', color='state', title="Childcare Cost vs. Taxes vs. Median Family Income by State")
fig15.write_html("C:\\PATH\\TO\\Final_Scatter3d15.html")

# 16)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_quant)
pca = PCA(n_components=3)
df_pca = pca.fit_transform(df_scaled)
plt.figure()
fig16 = px.scatter_3d(x=df_scaled[:, 0], y=df_scaled[:, 1], z=df_scaled[:, 2], title="3 Component PCA")
fig16.write_html("C:\\PATH\\TO\\Final_pca.html")

# 17)
model = NMF(n_components=3, init='random', random_state=0)
W = model.fit_transform(df_quant)
H = model.components_
fig17 = px.scatter_3d(x=W[:, 0], y=W[:, 1], z=W[:, 2], title="3 Component NMF")
fig17.write_html("C:\\PATH\\TO\\Final_nmf.html")
