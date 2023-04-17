from EmoTFIDF import EmoTFIDF
comment = 'I just love getting stuck in traffic for hours'
emTFIDF = EmoTFIDF()
emTFIDF.set_text(comment)
print(emTFIDF.em_frequencies)
emTFIDF = EmoTFIDF()
corpus_context = ["Residents of the city have expressed frustration and concern over the recent comments made by a "
                  "local business owner who claimed to 'just love getting stuck in traffic for hours'. The statement, "
                  "which was made during an interview with a local newspaper, has sparked outrage among commuters who "
                  "regularly face long delays on the city's busy roads.", 'Many residents have taken to social media '
                                                                          'to express their anger at the insensitive '
                                                                          'remark, with comments like:', "- 'I can't "
                                                                                                         "believe "
                                                                                                         "someone "
                                                                                                         "could be so "
                                                                                                         "out of "
                                                                                                         "touch. "
                                                                                                         "Getting "
                                                                                                         "stuck in "
                                                                                                         "traffic is "
                                                                                                         "a "
                                                                                                         "nightmare, "
                                                                                                         "not "
                                                                                                         "something "
                                                                                                         "to be "
                                                                                                         "enjoyed.'\n"
                                                                                                         "- 'This "
                                                                                                         "just goes "
                                                                                                         "to show how "
                                                                                                         "little some "
                                                                                                         "people "
                                                                                                         "understand "
                                                                                                         "the "
                                                                                                         "struggles "
                                                                                                         "that "
                                                                                                         "commuters "
                                                                                                         "face every "
                                                                                                         "day. It's "
                                                                                                         "not a game, "
                                                                                                         "it's a "
                                                                                                         "serious "
                                                                                                         "problem "
                                                                                                         "that needs "
                                                                                                         "to be "
                                                                                                         "addressed"
                                                                                                         ".'\n- 'As "
                                                                                                         "someone who "
                                                                                                         "has to sit "
                                                                                                         "in traffic "
                                                                                                         "for hours "
                                                                                                         "every day, "
                                                                                                         "I find this "
                                                                                                         "comment "
                                                                                                         "incredibly "
                                                                                                         "insulting. "
                                                                                                         "It's not "
                                                                                                         "something "
                                                                                                         "to be joked "
                                                                                                         "about.'",
                  "Local officials have also condemned the statement, with the city's mayor calling it 'insensitive "
                  "and disrespectful' to residents who are forced to endure long commutes due to the city's "
                  "inadequate infrastructure.", 'The issue of traffic congestion has been a major concern for '
                                                'residents of the city in recent years, with many calling for '
                                                'improvements to be made to the road network and public '
                                                'transportation system. The comments made by the business owner have '
                                                'only served to highlight the growing frustration and dissatisfaction '
                                                'felt by many in the community.', 'As the debate over traffic '
                                                                                  'congestion continues, '
                                                                                  'residents are calling on local '
                                                                                  'leaders and businesses to take '
                                                                                  'action to address the issue and '
                                                                                  'improve the quality of life for '
                                                                                  'everyone in the city.']
emTFIDF.computeTFIDF(corpus_context)
emTFIDF.set_text(comment)
emTFIDF.get_emotfidf()
print(emTFIDF.em_tfidf)
