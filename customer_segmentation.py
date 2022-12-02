import streamlit as st
import pandas as pd
import streamlit_keboola_api.src.keboola_api as kb
import os
from io import StringIO
from kbcstorage.client import Client
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


client = Client(st.secrets.url, st.secrets.key) 

@st.experimental_memo(ttl=7200)
def read_df(table_id, index_col=None, date_col=None):
    client.tables.export_to_file(table_id, '.')
    table_name = table_id.split(".")[-1]

    return pd.read_csv(table_name, index_col=index_col, parse_dates=date_col)

def read_df_segment(table_id, index_col=None, date_col=None):
    client.tables.export_to_file(table_id, '.')
    table_name = table_id.split(".")[-1]

    return pd.read_csv(table_name, index_col=index_col, parse_dates=date_col)


def saveFile(uploaded):
    with open(os.path.join(os.getcwd(),uploaded.name),"w") as f:
        strIo= StringIO(uploaded.getvalue().decode("utf-8"))
        f.write(strIo.read())
        return os.path.join(os.getcwd(),uploaded.name)



def segment_f(row,segments):
    filter_1 = ((segments["order_val_min"]<=row["order_val"] )& (segments["order_val_max"]>=row["order_val"]))
    filter_2 = ((segments["recency_min"]<=row["recency"]) & (segments["recency_max"]>=row["recency"]))
    filter_3 = ((segments["n_purchases_min"]<=row["n_purchases"]) & (segments["n_purchases_max"]>=row["n_purchases"])) 
    value = (segments[ (filter_2) & (filter_1)  & (filter_3) ]["General_Segment"].unique())
    if value.size==0:
        return "No Segment"
    else:
        return (value[0])



df_customers = read_df('in.c-wine.customers')
df_orders = read_df('in.c-wine.wine_orders', date_col=["order_date"])
segments = read_df_segment("out.c-create_segments.segments")
df_customers.rename(columns={"days_since_last_purchase":"recency","average_order":"order_val"},inplace=True)
# Create Lables for Each RFM Metric:Create generator of values for labels with range function
df_customers["General_Segment"] = df_customers.apply(lambda x : segment_f(x, segments=segments),axis=1)

main_header = '<p style="font-family:sans-serif; color:#121212; font-size: 36px;">Customer Segmentator</p>'
st.markdown(main_header, unsafe_allow_html=True)
st.markdown("#")
col_1,col_2,col_3,col_4 = st.columns(4)    
col_1.metric("Total Customers", df_customers.shape[0])
col_2.metric("Average Order Value", f"${df_customers.order_val.mean():.0f}")
col_3.metric("Average Number of Orders", f"{df_customers.n_purchases.mean():.0f}")
col_4.metric("Average days between Orders", f"{df_customers.recency.mean():.0f}")
st.markdown("#")

subheader_1 = '<p style="font-family:sans-serif; color:#1f1f1f; font-size: 28px;">Overview of Segments</p>'
st.markdown(subheader_1, unsafe_allow_html=True)
summary = df_customers.groupby('General_Segment').agg({
'recency':'mean',
'order_val' :'mean',
'n_purchases':['mean','count']
    })
summary.columns = summary.columns.map('_'.join)
summary = summary.reset_index()
st.dataframe(summary)
st.markdown("#")


#color_dict = {"Lost Sheep" : "red","Need Attention":"orange","Potentially Loyal":"blue","Loyal":"light green","MVC":"green"}
fig = sns.relplot(data=df_customers, x='recency', y='n_purchases', hue='General_Segment', height=8.27, aspect=11.7/8.27)
sns.move_legend(fig, "upper right")
st.pyplot(fig)

subheader_2 = '<p style="font-family:sans-serif; color:#1f1f1f; font-size: 28px;">Playground</p>'
st.markdown(subheader_2, unsafe_allow_html=True)
st.markdown("#")
col6,col7,col8 = st.columns(3)
st.markdown("#")
col1, col2, col3, col4_ = st.columns(4)
st.markdown("#")
col4,col5 = st.columns(2)

customers, orders = st.tabs(["Customers", "Orders"])
st.markdown("#")

subheader_3 = '<p style="font-family:sans-serif; color:#1f1f1f; font-size: 28px;">Evaluate Segments</p>'
st.markdown(subheader_3, unsafe_allow_html=True)
with col6:
    n_past_purchases_low, n_past_purchases_high = st.slider(
            'Number of orders',
            1, int(df_customers.n_purchases.max()), (1, int(df_customers.n_purchases.max())))
    
with col7:
    n_days_since_order_low, n_days_since_order_high = st.slider(
            'Number of days since last order',
            1, int(df_customers.recency.max()), (1, int(df_customers.recency.max())))
with col8:
    consumer_basket_low, consumer_basket_high = st.slider(
            'Average order value',
            0, int(df_customers.order_val.max()), (0, int(df_customers.order_val.max())))

filter_customers_purchases = (df_customers.n_purchases>=n_past_purchases_low) & (df_customers.n_purchases<=n_past_purchases_high)
filter_customers_consumer_basket = (df_customers.order_val>=consumer_basket_low) & (df_customers.order_val<=consumer_basket_high)
filter_customers_days = (df_customers.recency>=n_days_since_order_low) & (df_customers.recency<=n_days_since_order_high)
filter_customers =  filter_customers_purchases & filter_customers_consumer_basket & filter_customers_days

df_customers_filtered = df_customers.loc[filter_customers]
df_orders_filtered = df_orders.loc[df_orders.customer_id.isin(df_customers_filtered.index)]




col1.metric("Number of Customers", df_customers_filtered.shape[0])
col2.metric("Average Order Value", f"${df_customers_filtered.order_val.mean():.2f}")
col3.metric("Average Number of Orders", f"{df_customers_filtered.n_purchases.mean():.0f}")
col4_.metric("Average days between Orders", f"{df_customers_filtered.recency.mean():.0f}")


with col4:
    st.text("Distribution of Order Values in $")
    fig, ax = plt.subplots()
    ax.hist(df_customers_filtered.order_val, bins=5)
    st.pyplot(fig)

with col5:
    st.text("% of customers by number of orders")
    fig1, ax1 = plt.subplots()
    df_customers_filtered.n_purchases.value_counts().plot(kind="pie",autopct='%1.0f%%')
    #ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    #        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)

with customers:
    st.dataframe(df_customers_filtered)

with orders:
    st.dataframe(df_orders_filtered)

st.markdown("Choose from the below options if you would like to edit existing segments or add new segments")


with st.expander("Existing segments"): 
    gd_1 = GridOptionsBuilder.from_dataframe(segments)
    gd_1.configure_default_column(editable=True,groupable=True)
    gd_1.configure_selection(selection_mode="multiple", use_checkbox=True)
    gridoptions = gd_1.build()
    grid_table_1 = AgGrid(segments,gridOptions=gridoptions,
                        update_mode= GridUpdateMode.VALUE_CHANGED | GridUpdateMode.SELECTION_CHANGED,
                        height = 100,
                        allow_unsafe_jscode=True,
                        enable_enterprise_modules=False

    )
    sel_row_1 = pd.DataFrame(grid_table_1["selected_rows"])
    if sel_row_1.size:
        sel_row_1= sel_row_1[["General_Segment","recency_min","recency_max","n_purchases_min","n_purchases_max","order_val_min","order_val_max"]]
        sel_row_1.to_csv("test_1.csv",index=False)
        
        value = kb.keboola_create_update(keboola_URL=client.root_url, 
                                    keboola_key=client._token, 
                                    keboola_table_name="segments", 
                                    keboola_bucket_id="out.c-create_segments", 
                                    keboola_file_path="test_1.csv", 
                                    keboola_is_incremental = True,
                                    keboola_primary_key="segment_name",
                                    #Button Label
                                    label="SAVE SEGMENT",
                                    # Key is mandatory and has to be unique
                                    key="two",
                                    # if api_only= True than the button is not shown and the api call is fired directly
                                    api_only=False
                                    )
        value

with st.expander("New segment"): 
    segments_new = pd.DataFrame([["",n_purchases_min, n_purchases_max,recency_min, recency_max,order_val_min,order_val_max]],columns = ["General_Segment","n_purchases_min","n_purchases_max","recency_min","recency_max","order_val_min","order_val_max"])
    gd = GridOptionsBuilder.from_dataframe(segments_new)
    gd.configure_default_column(editable=True,groupable=True)
    gd.configure_selection(selection_mode="multiple", use_checkbox=True)
    gridoptions = gd.build()
    grid_table = AgGrid(segments_new,gridOptions=gridoptions,
                        update_mode= GridUpdateMode.VALUE_CHANGED | GridUpdateMode.SELECTION_CHANGED,
                        height = 100,
                        allow_unsafe_jscode=True,
                        enable_enterprise_modules=False
    )

    sel_row = pd.DataFrame(grid_table["selected_rows"])
    if sel_row.size:
        sel_row= sel_row[["General_Segment","recency_min","recency_max","n_purchases_min","n_purchases_max","order_val_min","order_val_max"]]
        sel_row.to_csv("test.csv",index=False)
        value1 = kb.keboola_create_update(keboola_URL=client.root_url, 
                                    keboola_key=client._token, 
                                    keboola_table_name="segments", 
                                    keboola_bucket_id="out.c-create_segments", 
                                    keboola_file_path="test.csv", 
                                    keboola_is_incremental = True,
                                    keboola_primary_key="segment_name",
                                    #Button Label
                                    label="SAVE SEGMENT",
                                    # Key is mandatory and has to be unique
                                    key="three",
                                    # if api_only= True than the button is not shown and the api call is fired directly
                                    api_only=False
                                    )
        value1

