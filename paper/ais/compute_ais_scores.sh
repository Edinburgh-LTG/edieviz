#!/bin/bash

# FP
echo -e "True Positives"
cat $1 | lxgrep -w l "document[meta[attr[@name='goldlabel' and .='1']] and text[.//s[.//ent[@type~'ischaem' and not(@neg)] and .//ent[not(.~'[Ss]ub[-]?acute') and @stime~'recent' and not(@neg)]]]]" | lxcount -q document | grep 'Total'
# FN
echo -e "False Negatives"
cat $1 | lxgrep -w l "document[meta[attr[@name='goldlabel' and .='1']] and not(text[.//s[.//ent[@type~'ischaem' and not(@neg)] and .//ent[not(.~'[Ss]ub[-]?acute') and @stime~'recent' and not(@neg)]]])]" | lxcount -q document | grep 'Total'
# FP
echo -e "False Positives"
cat $1 | lxgrep -w l "document[meta[attr[@name='goldlabel' and .='0']] and text[.//s[.//ent[@type~'ischaem' and not(@neg)] and .//ent[not(.~'[Ss]ub[-]?acute') and @stime~'recent' and not(@neg)]]]]" | lxcount -q document | grep 'Total'
# TN
echo -e "True Negatives"
cat $1 | lxgrep -w l "document[meta[attr[@name='goldlabel' and .='0']] and not(text[.//s[.//ent[@type~'ischaem' and not(@neg)] and .//ent[not(.~'[Ss]ub[-]?acute') and @stime~'recent' and not(@neg)]]])]" | lxcount -q document | grep 'Total'
