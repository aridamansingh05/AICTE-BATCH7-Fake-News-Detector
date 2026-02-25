import textstat

def readability_score(text):
    return textstat.flesch_reading_ease(text)

def credibility_score(prediction_prob, readability):
    score = prediction_prob * 70

    if readability < 30:
        score -= 10

    return round(score, 2)
