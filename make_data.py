import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('data/DSV_DHTD_full.csv')
df.columns = df.iloc[0]
df = df[1:]

# # write column to file 
# with open("columns.txt", "w") as file:
#     for idx, column in enumerate(df.columns, start=1):
#         file.write(f"{idx} - {column}\n")

#chage column
number_to_term = {
    1: "StudentID", 2: "Gender", 3: "ParentEdu", 4: "PartTimeJob", 5: "FinancialSupport",
    6: "StudyTime", 7: "SocialMediaTime", 8: "SocialMediaUse", 9: "HealthStatus", 10: "MiddleSchoolGPA",
    11: "MathScore", 12: "LitScore", 13: "EngScore", 14: "AdmissionScore", 15: "HistoryScore",
    16: "CivicEdScore", 17: "PhysicsScore", 18: "ChemScore", 19: "BioScore", 20: "HighSchoolGPA",
    21: "EntryEngScore", 22: "AdmissionMethod", 23: "AdmissionPriority", 24: "Scholarship", 25: "Adaptability",
    26: "StudyMethod", 27: "UnivSupport", 28: "FacultySupport", 29: "FacilityQuality", 30: "FacultyQuality",
    31: "CurriculumRelevance", 32: "Competitiveness", 33: "PeerInfluence", 34: "MotivationSurvey", 35: "MajorMotivation",
    36: "LinearAlg", 37: "Calc1", 38: "AnalyticGeom", 39: "Psychology", 40: "Marxism1",
    41: "Informatics", 42: "Calc2", 43: "Pedagogy", 44: "Marxism2", 45: "English",
    46: "Elective3", 47: "TeachingSkills1", 48: "RevolutionHistory", 49: "GenAlg", 50: "Calc3",
    51: "MathTeachMeth", 52: "PracticalVietnamese", 53: "Elective", 54: "AffineEuclidGeom", 55: "Arithmetic",
    56: "TeachingSkills2", 57: "HoChiMinhIdeology", 58: "ComplexFunc", 59: "ProjectiveGeom", 60: "NumberTheory",
    61: "TeachingSkills3", 62: "TeachInternship1", 63: "GenTopology", 64: "ElemAlg", 65: "MeasureTheory",
    66: "ProbStats", 67: "DiffEq", 68: "MathContent", 69: "SpecEnglish", 70: "FuncAnalysis",
    71: "GenLaw", 72: "PDEq", 73: "PubAdmin", 74: "LinearProgramming", 75: "TeachInternship2",
    76: "NumericalAnalysis", 77: "ElemGeom", 78: "MathInEnglish", 79: "ResearchMeth", 80: "Elective7",
    81: "Elective8", 82: "Elective9", 83: "TeachInternship3", 84: "Thesis", 85: "DiffInstrAlg",
    86: "DiffInstrGeom", 87: "AdvSeqFunc", 88: "Classification"
}

with open("columns.txt", "w") as file:
    for idx, column in enumerate(df.columns, start=1):
        file.write(f"{idx} - {number_to_term[idx]} - {column}\n")
        
# Replace column names using the dictionary
df.columns = [number_to_term[idx] if (idx) in number_to_term else col 
              for idx, col in enumerate(df.columns,start=1)]

# chage lable Classification
ordinal_encoder = OrdinalEncoder(categories=[['Trung bình','Khá', 'Giỏi', 'Xuất sắc', ]])
df['Classification'] = ordinal_encoder.fit_transform(df[['Classification']])

df = df.astype('float64')
pd.set_option('display.precision', 16)


df.to_csv('DSV_DHTD_without_index.csv', index=False)

