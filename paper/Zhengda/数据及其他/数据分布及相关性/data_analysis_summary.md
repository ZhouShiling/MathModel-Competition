# Data Distribution and Correlation Analysis

This document summarizes the data distribution and correlation analysis for the AI Capability Analysis project.

## Data Overview

The dataset includes responses from 250 graduate students, with the following key variables:

- Demographic information (age, gender, academic background)
- AI knowledge assessment (5 dimensions, 20 items)
- Technical skills assessment (4 dimensions, 16 items)
- Problem-solving ability assessment (4 dimensions, 16 items)
- Learning attitude assessment (4 dimensions, 16 items)
- AI capability assessment (4 dimensions, 16 items)

## Descriptive Statistics

### AI Knowledge

| Dimension | Mean | SD | Min | Max |
|-----------|------|----|----|-----|
| Machine Learning | 3.75 | 0.82 | 1.0 | 5.0 |
| Deep Learning | 3.42 | 0.95 | 1.0 | 5.0 |
| AI Applications | 3.88 | 0.76 | 1.5 | 5.0 |
| AI Ethics | 3.56 | 0.89 | 1.0 | 5.0 |
| Overall | 3.65 | 0.72 | 1.2 | 5.0 |

### Technical Skills

| Dimension | Mean | SD | Min | Max |
|-----------|------|----|----|-----|
| Programming | 3.62 | 0.94 | 1.0 | 5.0 |
| Data Preprocessing | 3.78 | 0.85 | 1.0 | 5.0 |
| Model Implementation | 3.45 | 0.98 | 1.0 | 5.0 |
| Tool Usage | 3.82 | 0.79 | 1.5 | 5.0 |
| Overall | 3.67 | 0.81 | 1.1 | 5.0 |

### Problem-Solving Ability

| Dimension | Mean | SD | Min | Max |
|-----------|------|----|----|-----|
| Problem Formulation | 3.58 | 0.87 | 1.0 | 5.0 |
| Solution Design | 3.49 | 0.92 | 1.0 | 5.0 |
| Implementation | 3.62 | 0.88 | 1.0 | 5.0 |
| Evaluation | 3.71 | 0.83 | 1.5 | 5.0 |
| Overall | 3.60 | 0.79 | 1.2 | 5.0 |

### Learning Attitude

| Dimension | Mean | SD | Min | Max |
|-----------|------|----|----|-----|
| Motivation | 4.12 | 0.76 | 1.5 | 5.0 |
| Self-efficacy | 3.85 | 0.82 | 1.0 | 5.0 |
| Persistence | 3.92 | 0.79 | 1.5 | 5.0 |
| Curiosity | 4.05 | 0.74 | 1.5 | 5.0 |
| Overall | 3.98 | 0.68 | 1.6 | 5.0 |

### AI Capability

| Dimension | Mean | SD | Min | Max |
|-----------|------|----|----|-----|
| Project Performance | 3.68 | 0.86 | 1.0 | 5.0 |
| Knowledge Application | 3.72 | 0.83 | 1.0 | 5.0 |
| Innovation | 3.54 | 0.92 | 1.0 | 5.0 |
| Practical Impact | 3.62 | 0.88 | 1.0 | 5.0 |
| Overall | 3.64 | 0.79 | 1.2 | 5.0 |

## Correlation Analysis

### Correlation Matrix (Overall Dimensions)

|                       | AI Knowledge | Technical Skills | Problem-Solving | Learning Attitude | AI Capability |
|-----------------------|--------------|------------------|----------------|-------------------|---------------|
| AI Knowledge          | 1.00         | 0.65             | 0.58           | 0.42              | 0.72          |
| Technical Skills      | 0.65         | 1.00             | 0.68           | 0.38              | 0.68          |
| Problem-Solving       | 0.58         | 0.68             | 1.00           | 0.45              | 0.64          |
| Learning Attitude     | 0.42         | 0.38             | 0.45           | 1.00              | 0.52          |
| AI Capability         | 0.72         | 0.68             | 0.64           | 0.52              | 1.00          |

All correlations are significant at p < 0.01.

## Key Findings

1. **AI Knowledge and Technical Skills** have the strongest correlation with AI Capability (r = 0.72 and r = 0.68, respectively).

2. **Technical Skills and Problem-Solving Ability** are strongly correlated (r = 0.68), suggesting that technical skills contribute to problem-solving ability.

3. **Learning Attitude** has a moderate correlation with AI Capability (r = 0.52), indicating that motivation and persistence play a role in developing AI capability.

4. **AI Knowledge and Technical Skills** are strongly correlated (r = 0.65), suggesting that theoretical knowledge and practical skills develop together.

5. The distribution of scores across all dimensions is slightly skewed toward the higher end, with means ranging from 3.45 to 4.12 on a 5-point scale.

## Implications

1. Training programs should focus on both theoretical knowledge and practical skills to maximize AI capability development.

2. Problem-solving exercises should be integrated with technical skill development.

3. Fostering positive learning attitudes can enhance the development of AI knowledge and capability.

4. The strong correlations between dimensions support the integrated approach to AI capability development proposed in the research model.