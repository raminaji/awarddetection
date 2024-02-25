import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
nlp = spacy.load("en_core_web_sm")
print("core web sm loaded")

data = [
    ("Race: White, Black, African American, Indian, Chinese, Alaskan, Asian, Arabic, Middle Eastern, Mexican, Hispanic, Japanese, Vietnamese", "Race"),
    ("Awards: Science Fair Winner, National Merit Scholar", "Awards"),
]

X = [" ".join([token.lemma_ for token in nlp(text)]) for text, _ in data]
y = [label for _, label in data]
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
print("data vectorized")
classifier = LinearSVC()
classifier.fit(X_vectorized, y)
application='''
Chance a surprised Georgetown and ND reject for Ivy’s and others
Demographics Race:White Male Citizenship: USA and Hungary Residence: Pennsylvania, USA Income: Full Pay School Type: Private Jesuit Intended Major: Political Science Hooks: Legacy at Princeton, UPenn, and Holy Cross
Academics GPA: 3.87 UW(3.91 without A-s)/ 4.2W Coursework: 8APs, 9 Honors by graduation Rank: Top 10% Testing: SAT 1590(800M/790R) AP Human Geography 5, APUSH 3(not sending), APGOV 4, AP Lang 5
ECs:(names excluded to avoid doxing) 1. Student Body President:
Elected by student body of =950; organized events (dances, trips); directed fundraising efforts; created surveys to assess students’ mental health
2: Varsity Rower : Rowed year round for nationally ranked team, including a national championship
3.Intern, Office of My Congresswomen: Applied and selected as intern; worked with Congresswomen ___ to deign/critique policy proposals and speeches on voting rights/gun violence
Researcher and Presenter- Self driven project:
Worked in urban gardens/food banks; researched food injustice to develop proposal on reducing food insecurity; presented to Philadelphia City Council
Peer tutor/English Tutor/Worker
tutored classmates in "academic advantage" program; helped teach English to Afghan refugees; packed and delivered food to Philadelphia residents
Yale Young Global Scholars
Attended selective (=18% admitted) 2-week academic program on Politics, Law, & Economics; completed capstone research project on Chinese diplomacy
Intern at PwC Sydney Office
Analyzed Tesla's global market position; conducted SWOT analysis; presented findings and recommendations to Tesla representative
Camp Counselor at Native American school
Organized and carried out activities for a summer camp program in a Navajo community for children ages 8-12; also assisted with light labor projects
Pre-college programs: Attended selective summer academic programs on "International Relations" and "Washington and the World" at Oxford and Georgetown, respectively
Co founder of a digital magazine
1 of 2 co-creators of____, a new media magazine with student writers from 12 countries that seeks to promote cultural understanding
School list: Georgetown(REA)- Deferred Notre Dame (REA)-Rejected Villanova University UNC UVA Umich UPenn(double legacy) Rice University WashU Yale Princeton(legacy) Williams Amherst Middlebury (maybe ED2) NYU stern (Maybe ED2) Dartmouth College of the Holy Cross Boston College Northeastern Emory
Any feedback would be appreciated. Thanks :-)
'''
for ff in application.split(" "):
    new_text = ff
    new_text_processed = " ".join([token.lemma_ for token in nlp(new_text)])
    new_text_vectorized = vectorizer.transform([new_text_processed])
    predicted_label = classifier.predict(new_text_vectorized)
    if (str(predicted_label) == "['Race']"):
        print(new_text)
