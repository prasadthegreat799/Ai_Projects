import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


resumePath='resume.pdf'
openResume=open(resumePath,'rb')
pdfReader=PyPDF2.PdfFileReader(openResume)
pageHandle=pdfReader.getPage(0)

resumeText = pageHandle.extractText()
resumeText = resumeText.replace('o ','')
resumeText = resumeText.replace('|', '')


jobDesctiptionPath='DeveloperJobDescription.pdf'
openJobDescription=open(jobDesctiptionPath,'rb')
pdfReaderJob=PyPDF2.PdfFileReader(openJobDescription)
pageHandleJob=pdfReaderJob.getPage(0)

resumeTextJob = pageHandleJob.extractText()
resumeTextJo = resumeTextJob.replace('o ','')
resumeTextJob = resumeTextJob.replace('|', '')


text=[resumeText,resumeTextJob]

cv=CountVectorizer()

countMatrix=cv.fit_transform(text)



#match percentage
matchPercentage=cosine_similarity(countMatrix)[0][1]* 100
matchPercentage=round(matchPercentage,2)


print("\n\nYour Resume Matches about " + str(matchPercentage) +"% of the job Description\n\n")


