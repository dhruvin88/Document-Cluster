# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:15:27 2018

@author: Dhruvin
"""
from utils import get_files, get_abstracts, get_idf, save

file_path = get_files('./Part1')
filenames,abstracts = get_abstracts(file_path)
idf,voc = get_idf(abstracts)

#save abtracts filenames and idf values
save('filenames.p',filenames)
save('abstracts.p', abstracts)
save('idf_values.p',idf)
save('voc_list.p',voc)

