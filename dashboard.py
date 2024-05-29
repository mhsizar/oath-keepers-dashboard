import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import streamlit as st

# Load the normalized messages data
file_path = 'data/messages_normalized.csv'
df = pd.read_csv(file_path, low_memory=False)

# Rename columns:
df.rename(columns={'u.username': 'Username', 'ts.$date': 'DateSent', 'msg': 'Message'}, inplace=True)

## Basic Analyses

# Top users by message count
top_users = df['Username'].value_counts().head(10).reset_index()
top_users.columns = ['Username', 'Message Count']
fig_top_users = px.bar(top_users, x='Username', y='Message Count')

# Number of messages by date
df['DateSent'] = pd.to_datetime(df['DateSent'])
messages_by_date = df.groupby(df['DateSent'].dt.date).size().reset_index(name='Message Count')
fig_messages_by_date = px.line(messages_by_date, x='DateSent', y='Message Count')

## Network Analysis of User Reactions on Messages

# Extract User Reactions from the dataframe for each Message
reactions_columns = [col for col in df.columns if col.startswith('reactions.')]
reactions_data = []

for col in reactions_columns:
    if df[col].notnull().any():
        message_senders = df[['Username', col]].dropna().rename(columns={'Username': 'Sender', col: 'Reactors'})
        for index, row in message_senders.iterrows():
            reactors = row['Reactors'].strip('[]').replace("'", "").split(', ')
            for reactor in reactors:
                reactions_data.append((reactor, row['Sender']))

# Create a DataFrame from the Reactions
reactions_df = pd.DataFrame(reactions_data, columns=['Reactor', 'Sender'])

# Calculate the number of reactions between each pair of users
reactions_weights = reactions_df.groupby(['Reactor', 'Sender']).size().reset_index(name='weight')

# Calculate the number of connections (distinct) for each user
connections_count = pd.concat([reactions_df['Reactor'], reactions_df['Sender']]).value_counts()

# Filter for top 50 users by connections
top_users_connections = connections_count.head(50).index

# Filter connections to only include the top 50 users
filtered_connections = reactions_weights[
    (reactions_weights['Reactor'].isin(top_users_connections)) & 
    (reactions_weights['Sender'].isin(top_users_connections))]

# Create a Network Graph with weights
G = nx.from_pandas_edgelist(filtered_connections, source='Reactor', target='Sender', edge_attr='weight')

# Calculate node degrees
node_degrees = dict(G.degree())
nx.set_node_attributes(G, node_degrees, 'Connections')

# Extract node positions
pos = nx.spring_layout(G, k=2.0, iterations=50)

# Create edge trace with color based on weights
edge_trace = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    weight = edge[2]['weight']
    edge_trace.append(go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        line=dict(width=0.5 + weight * 0.2, color=f'rgba(0, 0, 255, {min(weight / 10, 1)})'),
        hoverinfo='none',
        mode='lines'
    ))
    
# Create node trace with size based on connections
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='Viridis',
        size=[],
        color=[],
        colorbar=dict(
            thickness=15,
            title='User Connections',
            xanchor='left',
            titleside='right'
        )
    )
)

for node in G.nodes():
    x, y = pos[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    degree = G.nodes[node]['Connections']
    node_trace['marker']['size'] += tuple([10 + degree * 2]) 
    node_trace['marker']['color'] += tuple([degree]) 
    node_info = f"{node} (Connections: {degree})"
    node_trace['text'] += tuple([node_info])
    
# Create the network graph
fig_connections_network = go.Figure(data=edge_trace + [node_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        text="Hover over the nodes to see their degree",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

# Create Streamlit app
st.title('Oath Keepers Messages Dashboard')
st.subheader(':blue[Insights into OK User Interactions and Message Content by Mehedi Hasan Sizar]')
st.markdown('---')

# Introduction
st.markdown("""
            This dashboard provides insights into the Oath Keepers messages data. 
            Specifically, it includes insights into user interactions, message distribution over time, and word 
            usage within the messages dataset. Below, you will find various visualizations and tables that highlight 
            key aspects of the data, such as the most active users, the network of interactions between users, and the
            most common words used in the messages.
            """)

# Network Graph of User Interactions
st.markdown("""
            ### Network Graph of Top 50 Users and their Interactions
            
            This graph shows the interactions between the top 50 users (based on the number of connections). The nodes represent the users, and the edges represent the interactions (reaction) between them. The color and size of the nodes corresponds to the number of connections each user has, and the color and thickness of the edges represents the strength of the interaction between each pair of users. 
            """)
st.plotly_chart(fig_connections_network)

# Top Users by Message Count
st.markdown("""
            ### Top Users by Message Count
            
            The bar chart below shows the top 10 users based on the number of messages they have sent.
            """)
st.plotly_chart(fig_top_users)

# Number of Messages by Date
st.markdown("""
            ### Distribution of Messages Over Time
            
            It is evident from the line chart below that the number of messages sent everyday has decreased over time. The peak in the beginning of the timeline around second week of March, 2021 indicate a period of increased activity. According to [CNN](https://www.cnn.com/2021/03/09/politics/oath-keepers-capitol-riot-justice-department/index.html?target=_blank), this was the time when the court ruled that the Oath Keepers' founder was involved with the planning and execution of the US Capitol riot.
            """)
st.plotly_chart(fig_messages_by_date)

# Word Cloud of Messages
st.markdown("""
            ### Word Cloud of Messages
            
            The word cloud visualizes the most frequently used words in the messages. The size of each word is proportional to its frequency in the dataset. Common English stopwords have been removed from the analysis.
            """)
message_words = ' '.join(df['Message'].dropna().astype(str))
stopwords = set(STOPWORDS)
custom_stopwords = set(['http', 'https', 'www', 'com', 'org', 'net', 'html', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 
                        'bit', 'ly', 'twitter', 'facebook', 'youtube', 'youtu', 'be', 'get', 'now', 'good', '-', 
                        '--', 'morning', 's', 'u', 'do', 'your'])
stopwords = stopwords.union(custom_stopwords)
wordcloud = WordCloud(width=800, height=400, background_color ='white', stopwords=stopwords).generate(message_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Analysis Tables in Tabs
tabs = st.tabs(["Top Reactors", "Users Receiving Highest Reactions", "Strongest Connections", "Top Words in Messages"])

with tabs[0]:
    top_reactors = reactions_df['Reactor'].value_counts().head(10).reset_index()
    top_reactors.columns = ['Username', 'Reaction Count']
    top_reactors.index = range(1, 11)
    st.markdown("""
    ### Top Reactors

    This table lists the top 10 users who have reacted to the most messages. It provides insight into the most engaged users in terms of reactions.
    """)
    st.table(top_reactors)

with tabs[1]:
    top_reacted_users = reactions_df['Sender'].value_counts().head(10).reset_index()
    top_reacted_users.columns = ['Username', 'Reaction Count']
    top_reacted_users.index = range(1, 11)
    st.markdown("""
    ### Users Receiving Highest Reactions

    This table lists the top 10 users who have received the most reactions to their messages. It highlights the users whose messages have generated the most engagement from others.
    """)
    st.table(top_reacted_users)

with tabs[2]:
    reactions_df['Pair'] = reactions_df.apply(lambda row: tuple(sorted([row['Reactor'], row['Sender']])), axis=1)
    strongest_connections = reactions_df.groupby('Pair').size().reset_index(name='Interaction Count').sort_values(by='Interaction Count', ascending=False).head(10)
    strongest_connections[['User1', 'User2']] = pd.DataFrame(strongest_connections['Pair'].tolist(), index=strongest_connections.index)
    strongest_connections = strongest_connections[['User1', 'User2', 'Interaction Count']]
    strongest_connections.index = range(1, 11)
    st.markdown("""
    ### Strongest Connections

    This table shows the strongest connections between pairs of users based on the number of reactions exchanged.
    """)
    st.table(strongest_connections)

with tabs[3]:
    message_words_list = message_words.split()
    filtered_words = [word for word in message_words_list if word.lower() not in stopwords]
    top_words = pd.DataFrame(Counter(filtered_words).most_common(10), columns=['Word', 'Count'])
    top_words.index = range(1, 11)
    st.markdown("""
    ### Top Words in Messages

    This table lists the top 10 most frequently used words in the messages, excluding common stopwords.
    """)
    st.table(top_words)

# Conclusion
st.markdown("""
            
This dashboard provides a comprehensive analysis of the messages dataset, highlighting key aspects such as user activity, interaction networks, and common word usage. Not having access to the member information and the messages from before the Capitol riot, it is difficult to draw definitive conclusions about the nature of the interactions. However, the data provides valuable insights into the dynamics of the Oath Keepers group and the engagement levels of its members. Further analysis could involve sentiment analysis of the messages to understand the overall tone and sentiment of the conversations within the group, which can be cross-referenced with the emails data and external events to gain deeper insights into the group's activities and motivations.
""")

