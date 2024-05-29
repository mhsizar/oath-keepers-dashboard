# Oath keepers' Messages Dashboard

This dashboard provides insights into the Oath Keepers messages data. Specifically, it includes insights into user interactions, message distribution over time, and word usage within the messages dataset. Below, you will find various visualizations and tables that highlight key aspects of the data, such as the most active users, the network of interactions between users, and the most common words used in the messages. Not having access to the member information and the messages from before the Capitol riot, it is difficult to draw definitive conclusions about the nature of the interactions. However, the data provides valuable insights into the dynamics of the Oath Keepers group and the engagement levels of its members. Further analysis could involve sentiment analysis of the messages to understand the overall tone and sentiment of the conversations within the group, which can be cross-referenced with the emails data and external events to gain deeper insights into the group's activities and motivations.
\
\
To learn more about the Oath Keepers and their activities visit this [DDoS](https://ddosecrets.com/wiki/Oath_Keepers) page.

## Installation 
Clone the repository:
```
$ git clone https://github.com/mhsizar/oath-keepers-dashboard.git
$ cd oath-keepers-dashboard
```
Set up a virtual environment (optional but recommended):
```
$ python3 -m venv venv
$ source venv/bin/activate
```
Install the required packages:
```
$ pip install -r requirements.txt
```

## Usage
Run the StreamLit App to launch the dashboard application on your browser:
```
streamlit run dashboard.py
```
Network Graph of Top 50 Users and their Interactions:
![](https://github.com/mhsizar/oath-keepers-dashboard/blob/main/images/network_graph.png)
Top Users by Message Count:
![](https://github.com/mhsizar/oath-keepers-dashboard/blob/main/images/top_users.png)
Word Cloud from Messages:
<br>
![](https://github.com/mhsizar/oath-keepers-dashboard/blob/main/images/wordcloud.png)
