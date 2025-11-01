# Real-world case studies of data breaches

## Introduction

Data breaches in AI development can have far-reaching consequences, from financial losses to reputational damage and legal ramifications. Understanding real-world cases of data breaches provides valuable lessons for AI developers and organizations on the importance of robust data security practices.

In this reading, we will examine two significant data breaches that occurred due to vulnerabilities in handling training data.

By the end of this reading, you will be able to:

- Summarize the ramifications of data breaches, the errors that allowed them to happen, the estimated losses, and any policy changes that resulted.

---

## Case study 1: The Facebook-Cambridge Analytica scandal (2018)

### Overview

One of the most high-profile data breaches in recent history involved Facebook and the political consulting firm Cambridge Analytica. The breach came to light in 2018 when it was revealed that Cambridge Analytica had harvested the personal data of millions of Facebook users without their consent. This data was used to build psychographic profiles for political advertising purposes, raising significant concerns about privacy and data security.

### The identified error

The breach was enabled by a loophole in Facebook’s API that allowed third-party apps to collect data not only from users who consented but also from their friends who had not given explicit permission. The data collection was initially authorized for academic research purposes, but the data was later sold to Cambridge Analytica, violating Facebook’s policies.

### Ramifications

- **Public trust:** The scandal led to a massive loss of trust in Facebook, with users and regulators questioning the company’s commitment to data privacy.
- **Legal consequences:** Facebook faced multiple lawsuits and regulatory scrutiny worldwide. The Federal Trade Commission (FTC) fined the company $5 billion for privacy violations—the largest fine ever that the FTC imposed at the time.
- **Political impact:** The breach raised concerns about the role of data analytics in political campaigns, particularly regarding manipulation and voter influence.

### Estimated losses

- **Financial impact:** In addition to the $5 billion FTC fine, Facebook faced additional legal costs and settlements. The scandal also resulted in a temporary decline in the company’s stock price.
- **Reputational damage:** The long-term damage to Facebook’s reputation has been significant, with ongoing debates about data privacy and the ethical use of data in AI.

### Policy changes

- **API restrictions:** In response to the breach, Facebook significantly restricted access to its APIs, limiting the amount of data that third-party developers could collect.
- **Data privacy reforms:** Facebook introduced stricter data privacy controls, including more transparent user consent mechanisms and improved auditing processes for third-party apps.
- **Increased regulation:** The breach contributed to the development of stricter data privacy regulations, such as the General Data Protection Regulation in Europe and the California Consumer Privacy Act in the United States.

---

## Case study 2: The MyFitnessPal data breach (2018)

### Overview

In early 2018, Under Armour, the company behind the popular fitness app MyFitnessPal, announced a data breach that affected approximately 150 million user accounts. The breach exposed usernames, email addresses, and hashed passwords, raising concerns about the security of user data on popular mobile platforms.

### The identified error

The breach occurred due to vulnerabilities in the app’s data storage and encryption practices. Although the passwords were hashed, some were protected using the older SHA-1 algorithm, which is less secure and more susceptible to brute-force attacks. The breach exploited these weaknesses, allowing unauthorized access to sensitive user information.

### Ramifications

- **User trust:** The breach eroded user trust in MyFitnessPal and Under Armour, with many users questioning the security of their personal health and fitness data.
- **Data integrity:** Although financial information was not compromised, the breach highlighted the importance of securing even seemingly non-sensitive data, as it could be used in combination with other data for malicious purposes.

### Estimated losses

- **Financial impact:** Under Armour faced legal costs, potential settlements, and the costs of notifying affected users. The company’s stock price also experienced a temporary dip following the breach announcement.
- **User base impact:** Some users deleted their accounts or stopped using the app due to concerns about data security.

### Policy changes

- **Encryption and hashing standards:** Under Armour upgraded its encryption and hashing practices, moving away from SHA-1 to more secure algorithms such as bcrypt. The company also implemented stronger security protocols for storing and transmitting user data.
- **Enhanced security monitoring:** The breach prompted Under Armour to invest in enhanced security monitoring and incident response capabilities, including regular security audits and vulnerability assessments.
- **User education:** Under Armour increased efforts to educate users about the importance of password security, encouraging them to use strong, unique passwords and enabling two-factor authentication for additional protection.

---

## Conclusion

These two case studies underscore the critical importance of data security in AI development and beyond. Both the Facebook–Cambridge Analytica scandal and the MyFitnessPal data breach highlight how vulnerabilities in data handling can lead to significant financial, legal, and reputational consequences. They also demonstrate the necessity of proactive data security measures, including strong encryption, regular security audits, and strict access controls.

By learning from these real-world examples, organizations can better understand the potential risks associated with data breaches and take the necessary steps to protect their data and maintain user trust. As AI continues to evolve, ensuring the security of the data that powers these systems will remain a top priority for developers and organizations alike.