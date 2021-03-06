<!-- created like this:
cat /group/ltg/projects/DIS/ohsumed/ANNLT/dict/v.le | egrep '\(SUBCAT SC' | egrep -v 'PRT|PREP' | sed -r 's/^\(([^ ]+)/<lex word="\1"/' | sed -r 's/" .*SUBCAT SC\_/"><cat>/' | sed -r 's/\) .*$/<\/cat><\/lex>/' | egrep -v INF | cat ~/xmltop - ~/xmlbot | lxgrep body | lxreplace -q body -n "'lexicon'" > /afs/inf.ed.ac.uk/group/sync3/users/grover/pipeline/lib/chunk/controlverbs.lex 

then edited to merge duplicates -->

<lexicon>
<lex word="abide"><cat>ING</cat></lex>
<lex word="acknowledge"><cat>ING</cat></lex>
<lex word="act"><cat>NP</cat></lex>
<lex word="admit"><cat>ING</cat></lex>
<lex word="adore"><cat>ING</cat></lex>
<lex word="anticipate"><cat>ING</cat></lex>
<lex word="appear"><cat>AP</cat><cat>NP</cat></lex>
<lex word="appreciate"><cat>ING</cat></lex>
<lex word="attempt"><cat>ING</cat></lex>
<lex word="avoid"><cat>ING</cat></lex>
<lex word="became"><cat>AP</cat><cat>NP</cat></lex>
<lex word="become"><cat>AP</cat><cat>NP</cat></lex>
<lex word="began"><cat>ING</cat></lex>
<lex word="begin"><cat>ING</cat></lex>
<lex word="begrudge"><cat>ING</cat></lex>
<lex word="begun"><cat>ING</cat></lex>
<lex word="blush"><cat>AP</cat></lex>
<lex word="break"><cat>AP</cat></lex>
<lex word="bulk"><cat>AP</cat></lex>
<lex word="burn"><cat>AP</cat></lex>
<lex word="burnt"><cat>AP</cat></lex>
<lex word="came"><cat>AP</cat><cat>ING</cat></lex>
<lex word="cease"><cat>ING</cat></lex>
<lex word="chance"><cat>ING</cat></lex>
<lex word="come"><cat>AP</cat><cat>ING</cat></lex>
<lex word="commence"><cat>ING</cat></lex>
<lex word="confide"><cat>ING</cat></lex>
<lex word="consider"><cat>ING</cat></lex>
<lex word="contemplate"><cat>ING</cat></lex>
<lex word="continue"><cat>ING</cat></lex>
<lex word="countenance"><cat>ING</cat></lex>
<lex word="delay"><cat>ING</cat></lex>
<lex word="deny"><cat>ING</cat></lex>
<lex word="detest"><cat>ING</cat></lex>
<lex word="die"><cat>AP</cat><cat>NP</cat></lex>
<lex word="disclaim"><cat>ING</cat></lex>
<lex word="discontinue"><cat>ING</cat></lex>
<lex word="disdain"><cat>ING</cat></lex>
<lex word="dislike"><cat>ING</cat></lex>
<lex word="dread"><cat>ING</cat></lex>
<lex word="endure"><cat>ING</cat></lex>
<lex word="enjoy"><cat>ING</cat></lex>
<lex word="envisage"><cat>ING</cat></lex>
<lex word="escape"><cat>ING</cat></lex>
<lex word="evade"><cat>ING</cat></lex>
<lex word="favor"><cat>ING</cat></lex>
<lex word="favour"><cat>ING</cat></lex>
<lex word="fear"><cat>ING</cat></lex>
<lex word="feel"><cat>AP</cat><cat>NP</cat></lex>
<lex word="felt"><cat>AP</cat><cat>NP</cat></lex>
<lex word="finish"><cat>ING</cat></lex>
<lex word="forget"><cat>ING</cat></lex>
<lex word="forgot"><cat>ING</cat></lex>
<lex word="forgotten"><cat>ING</cat></lex>
<lex word="form"><cat>NP</cat></lex>
<lex word="forswear"><cat>ING</cat></lex>
<lex word="forswore"><cat>ING</cat></lex>
<lex word="forsworn"><cat>ING</cat></lex>
<lex word="freeze"><cat>AP</cat></lex>
<lex word="froze"><cat>AP</cat></lex>
<lex word="frozen"><cat>AP</cat></lex>
<lex word="funk"><cat>ING</cat></lex>

<lex word="get"><cat>AP</cat><cat>ING</cat><cat>PASS</cat></lex>
<lex word="go"><cat>AP</cat><cat>ING</cat></lex>
<lex word="gone"><cat>AP</cat><cat>ING</cat></lex>
<lex word="got"><cat>AP</cat><cat>ING</cat><cat>PASS</cat></lex>
<lex word="gotten"><cat>AP</cat><cat>ING</cat><cat>PASS</cat></lex>
<lex word="grew"><cat>AP</cat></lex>
<lex word="grow"><cat>AP</cat></lex>
<lex word="grown"><cat>AP</cat></lex>
<lex word="grudge"><cat>ING</cat></lex>
<lex word="hate"><cat>ING</cat></lex>
<lex word="help"><cat>ING</cat><cat>BSE</cat></lex>
<lex word="imagine"><cat>ING</cat></lex>
<lex word="justify"><cat>ING</cat></lex>
<lex word="keep"><cat>ING</cat><cat>AP</cat></lex>
<lex word="kept"><cat>ING</cat><cat>AP</cat></lex>
<lex word="lain"><cat>AP</cat></lex>
<lex word="lament"><cat>ING</cat></lex>
<lex word="lay"><cat>AP</cat><cat>ING</cat></lex>
<lex word="let"><cat>BSE</cat></lex>
<lex word="lie"><cat>AP</cat><cat>ING</cat></lex>
<lex word="like"><cat>ING</cat></lex>
<lex word="loathe"><cat>ING</cat></lex>
<lex word="look"><cat>AP</cat><cat>NP</cat></lex>
<lex word="loom"><cat>AP</cat></lex>
<lex word="love"><cat>ING</cat></lex>
<lex word="merit"><cat>ING</cat></lex>
<lex word="mind"><cat>ING</cat></lex>
<lex word="miss"><cat>ING</cat></lex>
<lex word="need"><cat>ING</cat></lex>
<lex word="neglect"><cat>ING</cat></lex>
<lex word="omit"><cat>ING</cat></lex>
<lex word="part"><cat>AP</cat><cat>NP</cat></lex>
<lex word="play"><cat>AP</cat><cat>NP</cat></lex>
<lex word="plead"><cat>AP</cat></lex>
<lex word="pled"><cat>AP</cat></lex>
<lex word="postpone"><cat>ING</cat></lex>
<lex word="practice"><cat>ING</cat></lex>
<lex word="practise"><cat>ING</cat></lex>
<lex word="prefer"><cat>ING</cat></lex>
<lex word="prove"><cat>AP</cat><cat>NP</cat></lex>
<lex word="proven"><cat>AP</cat><cat>NP</cat></lex>
<lex word="quit"><cat>ING</cat></lex>
<lex word="recall"><cat>ING</cat></lex>
<lex word="recollect"><cat>ING</cat></lex>
<lex word="regret"><cat>ING</cat></lex>
<lex word="relish"><cat>ING</cat></lex>
<lex word="remain"><cat>AP</cat><cat>NP</cat></lex>
<lex word="remember"><cat>ING</cat></lex>
<lex word="repent"><cat>ING</cat></lex>
<lex word="report"><cat>ING</cat></lex>
<lex word="require"><cat>ING</cat></lex>
<lex word="resent"><cat>ING</cat></lex>
<lex word="resist"><cat>ING</cat></lex>
<lex word="resume"><cat>ING</cat></lex>
<lex word="risk"><cat>ING</cat></lex>
<lex word="run"><cat>AP</cat></lex>
<lex word="save"><cat>ING</cat></lex>
<lex word="saw"><cat>ING</cat></lex>
<lex word="scorn"><cat>ING</cat></lex>
<lex word="see"><cat>ING</cat></lex>
<lex word="seem"><cat>AP</cat><cat>NP</cat></lex>
<lex word="seen"><cat>ING</cat></lex>
<lex word="shirk"><cat>ING</cat></lex>
<lex word="show"><cat>AP</cat></lex>
<lex word="showed"><cat>AP</cat></lex>
<lex word="shown"><cat>AP</cat></lex>
<lex word="smell"><cat>AP</cat></lex>
<lex word="smelt"><cat>AP</cat></lex>
<lex word="sound"><cat>AP</cat><cat>NP</cat></lex>
<lex word="stand"><cat>AP</cat><cat>ING</cat></lex>
<lex word="start"><cat>AP</cat><cat>ING</cat></lex>
<lex word="stay"><cat>AP</cat><cat>NP</cat></lex>
<lex word="stick"><cat>ING</cat></lex>
<lex word="stood"><cat>AP</cat><cat>ING</cat></lex>
<lex word="stop"><cat>ING</cat></lex>
<lex word="strike"><cat>AP</cat></lex>
<lex word="struck"><cat>AP</cat></lex>
<lex word="stuck"><cat>ING</cat></lex>
<lex word="take"><cat>AP</cat></lex>
<lex word="taken"><cat>AP</cat></lex>
<lex word="taste"><cat>AP</cat></lex>
<lex word="took"><cat>AP</cat></lex>
<lex word="try"><cat>ING</cat></lex>
<lex word="turn"><cat>AP</cat><cat>NP</cat></lex>
<lex word="visualise"><cat>ING</cat></lex>
<lex word="visualize"><cat>ING</cat></lex>
<lex word="want"><cat>ING</cat></lex>
<lex word="wax"><cat>AP</cat></lex>
<lex word="went"><cat>AP</cat><cat>ING</cat></lex>
</lexicon>
