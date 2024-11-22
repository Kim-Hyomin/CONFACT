init_guess_sys_prompt="You are InfoHuntGPT, a world-class AI assistant used by journalists to quickly build knowledge of new sources."

ext_prompt="Google search has revealed some new information:\n\n%s\n\n Update your background check for \"%s\" using the new information. Do NOT delete any information, but make ADDITIONS where necessary, using the new information. Most likely, you will just need to add an extra item to the itemized list you previously created. Make minimal edits, and only incorporate what is relevant. Begin your response with \"**Background check**\""

answer_sys_prompt="Please provide an answer in a sentence to the question: %s, based on provided evidence.\nEvidence:%s\n\nBegin your response with Answer:"

[
    """**Background check for The Guardian**
About
Launched in 1821, The Guardian is a British daily newspaper published in London, UK. Its original name is The Manchester Guardian, and cotton merchant John Edward Taylor founded it. In 1993 the Guardian Media Group acquired the Observer.

The paper focuses on politics, policy, business, and international relations. Their coverage includes News and Opinion, Sports, Culture, Lifestyle, Podcasts, and more.

Funded by / Ownership
The Guardian and its sister publication, the Sunday newspaper The Observer, are owned by Guardian Media Group plc (GMG). Scott Trust Limited was created in 1936 to ensure the editorial independence of the publications and owns Guardian Media Group plc (GMG). The Guardian states that “The Scott Trust is the sole shareholder in Guardian Media Group, and its profits are reinvested in journalism and do not benefit a proprietor or shareholders.” Donations and advertising fund the Guardian.

Analysis / Bias
The Guardian has always been a left-wing publication throughout its history, as they have stated in various articles.
Story selection favors the left but is generally factual. They utilize emotionally loaded headlines such as “The cashless society is a con – and big finance is behind it” and “Trump back-pedals on Russian meddling remarks after an outcry.” 

A 2014 Pew Research Survey found that 72% of The Guardian’s audience is consistently or primarily liberal, 20% Mixed, and 9% consistently or mostly conservative. This indicates that a more liberal audience strongly prefers the Guardian. 

Failed Fact Checks
* Is everything you think you know about depression wrong? – False
* Firms bidding for government contracts asked if they back Brexit. – False
* The proportion of lung cancer cases only diagnosed after a visit to an A&E ranges from 15% in Guildford and Waverley in Surrey to 56% in Tower Hamlets and Manchester. – Inaccurate

            """,
            """**Background check for The New York Times**
About
The New York Times (sometimes abbreviated to NYT) is an American daily newspaper, founded and continuously published in New York City since September 18, 1851, by The New York Times Company.
The New York Times was initially founded by U.S. journalist and politician Henry Jarvis Raymond and former banker George Jones. 

Funded by / Ownership
The Ochs-Sulzberger family controls the New York Times through Class B shares. Since 1967, the company has been listed on the New York Stock Exchange under the symbol NYT. Class B shares are those that are held privately. The owner and publisher of the New York Times are The New York Times Company, and the Chairman is Arthur Gregg “A.G.” Sulzberger, succeeding his father Arthur Ochs Sulzberger Jr. He is the sixth member of the Ochs/Sulzberger family to serve as publisher since Adolph Ochs purchased the newspaper in 1896.
Mark Thompson became president and chief executive officer of The New York Times Company in 2012. Advertising and subscription fees generate revenue.

Analysis / Bias
The News York Times’ coverage includes News (World News, National News, Business News), Opinion Pieces, Editorials, Arts, Movies, Theater, Travel, NYC Guide, Food, Home & Garden, and  Fashion & Style.

The NYT looks at the issues from a progressive perspective and is regarded as “liberal.” According to a Pew Research Centers’ media polarization report, “the ideological Placement of Each Source’s Audience” places the audience for the New York Times as “consistently liberal.” Further, since 1960 The New York Times has only endorsed Democratic Presidential Candidates. 

Failed Fact Checks

* “We have a host of issues associated with high B.M.I.s. But correlation doesn’t prove causation, and there’s a significant body of research showing that weight stigma and weight cycling can explain most if not all of the associations we see between higher weights and poor health outcomes.” – MOSTLY FALSE
* A political map circulated by Sarah Palin’s PAC incited Rep. Gabby Giffords’ shooting – FALSE            
            """
]