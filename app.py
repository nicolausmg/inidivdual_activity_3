import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt

st.set_page_config(layout="wide")

# ---------- Load and clean data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("car_sales_with_states.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.strftime('%b')
    df['Month'] = pd.Categorical(df['Month'], categories=[
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ], ordered=True)
    df['Price ($)'] = pd.to_numeric(df['Price ($)'], errors='coerce')
    return df.dropna(subset=['Price ($)'])

df = load_data()

# ---------- Sidebar filters ---------- 
st.sidebar.title("Filter Your View")

min_date, max_date = df['Date'].min(), df['Date'].max()
date_range = st.sidebar.slider( "Select Date Range", min_value=min_date.date(), max_value=max_date.date(), value=(min_date.date(), max_date.date()))
min_price, max_price = int(df['Price ($)'].min()), int(df['Price ($)'].max())
price_range = st.sidebar.slider("Car Price Range ($)", min_price, max_price, (min_price, max_price))

selected_gender = st.sidebar.multiselect("Gender", df['Gender'].unique(), default=list(df['Gender'].unique()))
selected_trans = st.sidebar.multiselect("Transmission", df['Transmission'].unique(), default=list(df['Transmission'].unique()))
selected_body_style = st.sidebar.multiselect("Body Style", df['Body Style'].dropna().unique(), default=list(df['Body Style'].dropna().unique()))
selected_make = st.sidebar.multiselect("Manufacturer", df['Company'].unique(), default=list(df['Company'].unique()))

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# ---------- Apply filters ----------
filtered_df = df[
    (df['Date'] >= start_date) &
    (df['Date'] <= end_date) &
    (df['Price ($)'].between(*price_range)) &
    (df['Gender'].isin(selected_gender)) &
    (df['Transmission'].isin(selected_trans)) &
    (df['Company'].isin(selected_make)) &
    (df['Body Style'].isin(selected_body_style))
]

# ---------- KPIs ----------
total_sales = filtered_df['Price ($)'].sum()
avg_price = filtered_df['Price ($)'].mean()
top_model = filtered_df.groupby('Model')['Price ($)'].sum().idxmax()
cars_sold = len(filtered_df)
auto = len(filtered_df[filtered_df['Transmission'] == 'Auto'])
manual = len(filtered_df[filtered_df['Transmission'] == 'Manual'])


st.title("Car Sales Dashboard")
k1, k2, k3, k4 = st.columns(4)
k1.metric("**Total Sales**", f"$ {total_sales/1e6:,.2f} M")
k2.metric("**Avg. Price**", f"$ {avg_price:,.0f}")
k3.metric("**Top Model**", top_model)
k4.metric("**Cars Sold**", f"{cars_sold:,}")

# ---------- Main dashboard tabs ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Sales trends", 
    "Top dealerships",
    "Model performance", 
    "Regional analysis", 
    "Key insights",
    "Chatbot"
])


# ----- Tab 1: Sales Trends ----- #
with tab1:
    st.subheader("Monthly Revenue Analysis")

    # Group monthly sales by Year/Month and calculate total revenue
    monthly_rev = (filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['Price ($)'].sum().reset_index().rename(columns={'Date': 'YearMonth', 'Price ($)': 'Revenue'}))

    monthly_rev['YearMonth'] = monthly_rev['YearMonth'].astype(str)
    monthly_rev = monthly_rev.sort_values('YearMonth')
    peak_month = monthly_rev.loc[monthly_rev['Revenue'].idxmax(), 'YearMonth']
    low_month = monthly_rev.loc[monthly_rev['Revenue'].idxmin(), 'YearMonth']
    avg_sales = monthly_rev['Revenue'].mean()
    recent_growth = ((monthly_rev['Revenue'].iloc[-1] - monthly_rev['Revenue'].iloc[-2]) / monthly_rev['Revenue'].iloc[-2]) * 100 if len(monthly_rev) > 1 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Month", peak_month)
    m2.metric("Lowest Month", low_month)
    m3.metric("Avg Monthly Revenue", f"${avg_sales:,.0f}")
    m4.metric("Recent Growth", f"{recent_growth:+.1f} %")
    st.info("Recent growth is calculated as the percentage change from the last month to the second last month.")
    
    c1, c2 = st.columns(2)
    with c1:
        # Sales volume chart
        st.subheader(" Monthly Sales Volume")
        monthly_count = (filtered_df.groupby(filtered_df['Date'].dt.to_period('M')).size().reset_index(name='Count'))
        monthly_count['YearMonth'] = monthly_count['Date'].astype(str)
        monthly_count = monthly_count.sort_values('YearMonth')
        monthly_count['MovingAvg'] = monthly_count['Count'].rolling(window=3).mean()

        line_base = alt.Chart(monthly_count).encode(x=alt.X('YearMonth:T', title='Year-Month'))
        line_sales = line_base.mark_line(point=True, color='steelblue').encode(y=alt.Y('Count:Q', title='Cars Sold'),tooltip=['YearMonth:T', 'Count'])
        line_avg = line_base.mark_line(strokeDash=[5, 5], color='orange').encode(y=alt.Y('MovingAvg:Q', title='3-Month Moving Avg'),tooltip=['YearMonth:T', alt.Tooltip('MovingAvg:Q', format='.1f')])

        combined_chart = (line_sales + line_avg).properties(
            width=800,
            height=400,
        ).interactive()

        st.altair_chart(combined_chart, use_container_width=True)
    with c2:
        # Monthly revenue chart
        st.markdown("### Total Sales by Year")
        sales_by_year = filtered_df.groupby('Year')['Price ($)'].sum().reset_index()
        bar_chart = alt.Chart(sales_by_year).mark_bar(color='skyblue').encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Price ($):Q', title='Revenue ($)'),
            tooltip=[alt.Tooltip('Year:O'), alt.Tooltip('Price ($):Q', format='$,.0f')]
        ).properties(
            width=700,
            height=400,
        )

        labels = bar_chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5
        ).encode(
            text=alt.Text('Price ($):Q', format='$,.0f')
        )

        st.altair_chart(bar_chart + labels, use_container_width=True)
    st.info("I used Altair for the charts, as seen in class, so feel free to hover over the points for more details.")



# ----- Tab 2: Top Dealerships ----- #
with tab2:
    st.subheader("Top 5 Performing Dealerships")

    dealer_perf = (filtered_df.groupby('Dealer_Name')['Price ($)'].sum().reset_index().sort_values(by='Price ($)', ascending=False))
    top5_dealers = dealer_perf.head(5)

    bar_chart = alt.Chart(top5_dealers).mark_bar().encode(
        x=alt.X('Price ($):Q', title='Total Revenue'),
        y=alt.Y('Dealer_Name:N', sort='-x', title='Dealer'),
        tooltip=['Dealer_Name:N', alt.Tooltip('Price ($):Q', format='$,.0f')],
        color=alt.Color('Dealer_Name:N', legend=None)
    ).properties(
        width=700,
        height=300
    ).interactive()

    labels = alt.Chart(top5_dealers).mark_text(
        align='left',
        baseline='middle',
        dx=3  
    ).encode(
        x='Price ($):Q',
        y=alt.Y('Dealer_Name:N', sort='-x'),
        text=alt.Text('Price ($):Q', format='$,.0f')
)
    final_chart = bar_chart + labels
    st.altair_chart(final_chart, use_container_width=True)

    st.subheader("Dealer Performance Explorer")
    st.info("From the top 5 dealerships seen above, you can select one to see detailed stats.")
    selected_dealer = st.selectbox("Select a dealership to view detailed stats:", top5_dealers['Dealer_Name'])
    dealer_data = filtered_df[filtered_df['Dealer_Name'] == selected_dealer]

    total_sales = dealer_data['Price ($)'].sum()
    total_units = dealer_data.shape[0]
    best_model = dealer_data['Model'].value_counts().idxmax()
    top_brand = dealer_data['Company'].value_counts().idxmax()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"${total_sales:,.0f}")
    k2.metric("Units Sold", f"{total_units:,}")
    k3.metric("Best-Selling Model", best_model)
    k4.metric("Top Brand", top_brand)

# ----- Tab 3: Model performance ----- #  
with tab3:
    s1, s2 = st.columns(2)
    with s1:
        st.subheader("Top 10 Models by Revenue")
        top_models = (filtered_df.groupby('Model')['Price ($)'].sum().nlargest(10).reset_index().sort_values('Price ($)', ascending=True))
        model_chart = alt.Chart(top_models).mark_bar().encode(
            x=alt.X('Price ($):Q', title='Revenue ($)'),
            y=alt.Y('Model:N', sort='-x'),
            tooltip=[alt.Tooltip('Model:N'), alt.Tooltip('Price ($):Q', format='$,.0f')],
            color=alt.Color('Model:N', legend=None)
        ).properties(
            width=700,
            height=400
        )

        model_labels = model_chart.mark_text(
            align='left',
            baseline='middle',
            dx=3
        ).encode(
            text=alt.Text('Price ($):Q', format='$,.0f')
        )

        st.altair_chart(model_chart + model_labels, use_container_width=True)
    with s2:
        st.subheader("Sales by Body Style")
        body_sales = ( filtered_df.groupby('Body Style')['Price ($)'].sum().reset_index().sort_values('Price ($)', ascending=False))

        body_chart = alt.Chart(body_sales).mark_bar().encode(
            x=alt.X('Price ($):Q', title='Revenue ($)'),
            y=alt.Y('Body Style:N', sort='-x'),
            tooltip=[alt.Tooltip('Body Style:N'), alt.Tooltip('Price ($):Q', format='$,.0f')],
            color=alt.Color('Body Style:N', legend=None)
        ).properties(
            width=700,
            height=350
        )

        body_labels = body_chart.mark_text(
            align='left',
            baseline='middle',
            dx=3
        ).encode(
            text=alt.Text('Price ($):Q', format='$,.0f')
        )

        st.altair_chart(body_chart + body_labels, use_container_width=True)

# ----- Tab 4: Regional analysis ----- #
with tab4:
    st.subheader("Total sales by U.S. state")
    st.info("Hover over the points to see the total sales in each state.")

    if 'StateCode' in filtered_df.columns:
        state_sales = filtered_df.groupby('StateCode')['Price ($)'].sum().reset_index()
        fig = px.scatter_geo(
            state_sales,
            locations='StateCode',
            locationmode='USA-states',
            size='Price ($)',
            color='Price ($)',
            hover_name='StateCode',
            scope='usa',
        )
        st.plotly_chart(fig, use_container_width=True)

# ----- Tab 5: Key insights ----- #
with tab5:
    st.subheader("Executive Summary")

    top_make = filtered_df.groupby('Company')['Price ($)'].sum().idxmax()
    top_region = filtered_df.groupby('StateCode')['Price ($)'].sum().idxmax()
    peak_month = filtered_df['Month'].value_counts().idxmax()

    st.markdown(f"""
    - **Top-Selling Model:** {top_model}
    - **Best Performing Region:** {top_region}
    - **Peak Sales Month:** {peak_month}
    """)

    st.success(f"**Insight:** Consider increasing inventory for models like **{top_model}** during **{peak_month}** in **{top_region}**.")

# ----- Tab 6: Simple Chatbot ----- #
with tab6:
    st.subheader("🤖 Ask the Dashboard")
    st.info("I build this chatbot using the theory from the Moodle examples (specifically advanced_chatbot.py). Type in lower case please!")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "How can I help you explore car sales data?"}
        ]

    # Process user input here so it doesn't jump tabs
    user_input = st.chat_input("Ask me about sales, models, dealerships...")

    # Display existing chat messages (must be inside tab!)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input:
        # Save + show user input
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Basic parsing
        prompt = user_input.lower()
        reply = ""
        chart_type = "none"

        # Responses
        if "sales" in prompt or "trend" in prompt:
            reply = "Here's the monthly sales trend:"
            chart_type = "trend"
        elif "top model" in prompt or "best model" in prompt:
            reply = f"The top-selling model is **{top_model}** with total revenue of ${filtered_df.groupby('Model')['Price ($)'].sum().max():,.0f}."
        elif "dealership" in prompt or "dealer" in prompt:
            top_dealer = filtered_df.groupby('Dealer_Name')['Price ($)'].sum().idxmax()
            reply = f"The top-performing dealership is **{top_dealer}**." 
        elif "clear" in prompt:
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Chat history cleared. How can I help you?"}
            ]
            st.rerun()
        else:
            reply = "Try asking about sales trends, top models, or dealerships."

        # Show bot response
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

            if chart_type == "trend":
                monthly = (
                    filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['Price ($)']
                    .sum()
                    .reset_index()
                )
                monthly['Date'] = monthly['Date'].astype(str)
                st.line_chart(monthly.rename(columns={'Date': 'Month', 'Price ($)': 'Revenue'}).set_index('Month'))