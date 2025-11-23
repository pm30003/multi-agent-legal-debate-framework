#!/usr/bin/env python3
"""
Legal Documents Setup Script
Creates the folder structure and legal documents for RAG system
"""

import os
from pathlib import Path

def create_legal_document_structure():
    """Create the folder structure for legal documents"""
    
    # Base directory
    base_dir = Path("legal_documents")
    
    # Create directories
    directories = [
        base_dir / "case_law",
        base_dir / "statutes", 
        base_dir / "regulations"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return base_dir

def create_legal_documents(base_dir):
    """Create all legal documents with content"""
    
    # Case Law Documents
    case_law_dir = base_dir / "case_law"
    
    # Kesavananda Bharati Case
    kesavananda_content = """Kesavananda Bharati Sripadagalvaru and Ors. v. State of Kerala and Anr.
AIR 1973 SC 1461, (1973) 4 SCC 225

SUPREME COURT OF INDIA
Civil Appeal Nos. 135-136 of 1970
Decided on: April 24, 1973

JUDGMENT:
BY THE COURT:

This case involves the fundamental question of the extent of Parliament's power to amend the Constitution under Article 368. The petitioner challenged the Kerala Land Reforms (Amendment) Act, 1969, which was placed in the Ninth Schedule by the Constitution (Twenty-ninth Amendment) Act, 1972.

BASIC STRUCTURE DOCTRINE:
The Court held by a majority that while Parliament has wide powers to amend the Constitution, it cannot alter the "basic structure" or "essential features" of the Constitution. The basic structure includes:

1. Supremacy of the Constitution
2. Republican and Democratic form of government
3. Secular character of the Constitution
4. Separation of powers between the legislature, executive and judiciary
5. Federal character of the Constitution
6. Unity and integrity of the nation
7. Welfare state (socio-economic justice)
8. Judicial review
9. Rule of law
10. Individual freedom and dignity
11. Parliamentary system
12. Independence of the judiciary

MAJORITY JUDGMENT:
Sikri, C.J. (for the majority):
"The Constitution is a precious heritage; therefore, you cannot destroy its identity. While Parliament has constituent power, it cannot use this power to destroy the basic elements or fundamental features of the Constitution."

"The word 'amendment' postulates that the old Constitution survives without loss of its identity despite the change and continues even though it has been subjected to alterations."

IMPACT AND SIGNIFICANCE:
1. This judgment established the basic structure doctrine as a cornerstone of Indian constitutional law
2. It limited Parliament's amending power while preserving constitutional democracy
3. It strengthened judicial review and the independence of the judiciary
4. It protected fundamental rights from being completely abolished
5. It ensured the Constitution's democratic and secular character remains intact

The basic structure doctrine ensures that the Constitution remains a living document that can adapt to changing circumstances while preserving its fundamental character and the values it embodies."""

    with open(case_law_dir / "kesavananda_bharati_case.txt", "w", encoding="utf-8") as f:
        f.write(kesavananda_content)
    
    # Maneka Gandhi Case
    maneka_content = """Maneka Gandhi v. Union of India
AIR 1978 SC 597, (1978) 1 SCC 248

SUPREME COURT OF INDIA
Writ Petition (Civil) No. 27 of 1978
Decided on: January 25, 1978

JUDGMENT:
Bhagwati, J.:

This case marks a revolutionary expansion in the interpretation of Article 21 of the Constitution of India and establishes the interconnectedness of fundamental rights.

FACTS:
Maneka Gandhi, a journalist and social activist, had her passport impounded by the Regional Passport Officer, New Delhi, under Section 10(3)(c) of the Passports Act, 1967, on the ground that it was "in the interest of the general public" that the passport should be impounded. No reasons were provided to her for this action, nor was she given an opportunity to be heard before the impoundment.

REVOLUTIONARY INTERPRETATION OF ARTICLE 21:
The Court held that Article 21 is not merely a procedural safeguard but embodies substantive rights. The "procedure established by law" must be:
- Just, fair and reasonable
- Not arbitrary, fanciful or oppressive
- In accordance with principles of natural justice

Justice Bhagwati observed:
"The procedure cannot be arbitrary, unfair or unreasonable. The procedure must be 'right and just and fair' and not arbitrary, fanciful or oppressive; otherwise, it would be no procedure at all and the requirement of Article 21 would not be satisfied."

RIGHT TO TRAVEL ABROAD:
The Court recognized that the right to travel abroad is part of personal liberty under Article 21:
"Personal liberty in Article 21 is of the widest amplitude and it covers a variety of rights which go to constitute the personal liberty of man and some of them have been raised to the status of distinct fundamental rights and given additional protection under Article 19."

INTERCONNECTEDNESS OF FUNDAMENTAL RIGHTS:
The Court established that fundamental rights are not mutually exclusive but form an integrated scheme:
"Articles 14, 19 and 21 are not mutually exclusive. They sustain, strengthen and nourish each other. Article 21 derives its life and substance from Articles 14 and 19."

GOLDEN TRIANGLE DOCTRINE:
The judgment established the "Golden Triangle" of Articles 14, 19, and 21:
- Article 14 (Equality): Any law that is arbitrary would violate Article 14
- Article 19 (Freedom): Restrictions must be reasonable and in public interest  
- Article 21 (Life and Liberty): Procedure must be just, fair and reasonable

This case continues to be cited as the cornerstone for expanding fundamental rights and ensuring that state power is exercised in accordance with constitutional values of justice, fairness, and reasonableness."""

    with open(case_law_dir / "maneka_gandhi_case.txt", "w", encoding="utf-8") as f:
        f.write(maneka_content)
    
    # Vishaka Case
    vishaka_content = """Vishaka v. State of Rajasthan
AIR 1997 SC 3011, (1997) 6 SCC 241

SUPREME COURT OF INDIA
Writ Petition (Criminal) No. 666-70 of 1992
Decided on: August 13, 1997

JUDGMENT:
Verma, CJ.:

This case establishes the fundamental right of women to work in an environment free from sexual harassment. The guidelines laid down herein are known as the Vishaka Guidelines.

FACTS:
Bhanwari Devi, a social worker in Rajasthan, was gang-raped as an act of social vengeance when she tried to prevent a child marriage in her village. The incident highlighted the vulnerability of working women and the absence of specific laws dealing with sexual harassment at workplace.

FUNDAMENTAL RIGHT TO WORK WITH DIGNITY:
The Court held that the right to work with dignity is a fundamental right of women under Articles 14, 15, 19(1)(g) and 21 of the Constitution:
"Sexual harassment of women at workplace is incompatible with the dignity and honour of women and needs to be eliminated. It is a violation of the rights guaranteed under Articles 14, 15 and 21 of the Constitution."

VISHAKA GUIDELINES:
The Court laid down comprehensive guidelines for prevention of sexual harassment:

1. Definition of Sexual Harassment:
   - Unwelcome sexually determined behaviour (whether directly or by implication)
   - Physical contact and advances
   - Demand or request for sexual favours
   - Sexually coloured remarks
   - Showing pornography
   - Any other unwelcome physical, verbal or non-verbal conduct of sexual nature

2. Preventive Steps:
   - Express prohibition of sexual harassment
   - Awareness programs
   - Display of guidelines prominently
   - Appropriate penalties in service rules

3. Complaints Committee:
   - Complaints committee headed by a woman
   - Half of the members should be women
   - Third party participation to prevent undue pressure
   - Confidential treatment of complaints

4. Complaint Mechanism:
   - Complaint to be made within reasonable period
   - Written complaint with details
   - Inquiry to be completed within reasonable time
   - Natural justice principles to be followed

5. Punishment:
   - Appropriate disciplinary action
   - Penalty proportionate to misconduct
   - Compensation to victim in appropriate cases

IMPACT:
This judgment:
1. Recognized sexual harassment as violation of fundamental rights
2. Provided interim guidelines until legislation was enacted
3. Made workplace sexual harassment legally cognizable
4. Led to the Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013
5. Established judicial legislation in absence of parliamentary law

The Vishaka Guidelines remained in force until the enactment of the Sexual Harassment of Women at Workplace Act, 2013, and continue to be relevant for interpretation of women's rights at workplace."""

    with open(case_law_dir / "vishaka_case.txt", "w", encoding="utf-8") as f:
        f.write(vishaka_content)
    
    # Statutes
    statutes_dir = base_dir / "statutes"
    
    # Constitution of India
    constitution_content = """THE CONSTITUTION OF INDIA

PREAMBLE
WE, THE PEOPLE OF INDIA, having solemnly resolved to constitute India into a SOVEREIGN SOCIALIST SECULAR DEMOCRATIC REPUBLIC and to secure to all its citizens:
JUSTICE, social, economic and political;
LIBERTY of thought, expression, belief, faith and worship;
EQUALITY of status and of opportunity;
and to promote among them all
FRATERNITY assuring the dignity of the individual and the unity and integrity of the Nation;

PART III - FUNDAMENTAL RIGHTS

Article 14. Equality before law
The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.

Article 15. Prohibition of discrimination on grounds of religion, race, caste, sex or place of birth
(1) The State shall not discriminate against any citizen on grounds only of religion, race, caste, sex, place of birth or any of them.

Article 19. Protection of certain rights regarding freedom of speech, etc.
(1) All citizens shall have the right—
(a) to freedom of speech and expression;
(b) to assemble peaceably and without arms;
(c) to form associations or unions;
(d) to move freely throughout the territory of India;
(e) to reside and settle in any part of the territory of India; and
(g) to practise any profession, or to carry on any occupation, trade or business.

Article 21. Protection of life and personal liberty
No person shall be deprived of his life or personal liberty except according to procedure established by law.

Article 21A. Right to education
The State shall provide free and compulsory education to all children of the age of six to fourteen years in such manner as the State may, by law, determine.

Article 22. Protection against arrest and detention in certain cases
(1) No person who is arrested shall be detained in custody without being informed, as soon as may be, of the grounds for such arrest nor shall he be denied the right to consult, and to be defended by, a legal practitioner of his choice.

PART IV - DIRECTIVE PRINCIPLES OF STATE POLICY

Article 38. State to secure a social order for the promotion of welfare of the people
(1) The State shall strive to promote the welfare of the people by securing and protecting as effectively as it may a social order in which justice, social, economic and political, shall inform all the institutions of the national life.

Article 39. Certain principles of policy to be followed by the State
The State shall, in particular, direct its policy towards securing—
(a) that the citizens, men and women equally, have the right to an adequate means of livelihood;
(b) that the ownership and control of the material resources of the community are so distributed as best to subserve the common good;

Article 44. Uniform civil code for the citizens
The State shall endeavour to secure for the citizens a uniform civil code throughout the territory of India."""

    with open(statutes_dir / "constitution_of_india.txt", "w", encoding="utf-8") as f:
        f.write(constitution_content)
    
    # Indian Penal Code
    ipc_content = """THE INDIAN PENAL CODE, 1860
ACT NO. 45 OF 1860

PREAMBLE
An Act to provide a general Penal Code for India.

CHAPTER I - INTRODUCTION

Section 1. Title and extent of operation of the Code
This Act shall be called the Indian Penal Code, and shall extend to the whole of India except the State of Jammu and Kashmir.

Section 2. Punishment of offences committed within India
Every person shall be liable to punishment under this Code and not otherwise for every act or omission contrary to the provisions thereof, of which he shall be guilty within India.

CHAPTER III - OF PUNISHMENTS

Section 53. Punishments
The punishments to which offenders are liable under the provisions of this Code are—
First.—Death.
Secondly.—Imprisonment for life.
Thirdly.—Imprisonment, which is of two descriptions, namely:—
(1) Rigorous, that is, with hard labour.
(2) Simple.
Fourthly.—Forfeiture of property.
Fifthly.—Fine.

CHAPTER IV - GENERAL EXCEPTIONS

Section 76. Act done by a person bound, or by mistake of fact believing himself bound, by law
Nothing is an offence which is done by a person who is, or who by reason of a mistake of fact and not by reason of a mistake of law in good faith believes himself to be, bound by law to do it.

Section 82. Act of a child under seven years of age
Nothing is an offence which is done by a child under seven years of age.

Section 84. Act of a person of unsound mind
Nothing is an offence which is done by a person who, at the time of doing it, by reason of unsoundness of mind, is incapable of knowing the nature of the act, or that he is doing what is either wrong or contrary to law.

CHAPTER XVI - OF OFFENCES AFFECTING THE HUMAN BODY

Section 299. Culpable homicide
Whoever causes death by doing an act with the intention of causing death, or with the intention of causing such bodily injury as is likely to cause death, or with the knowledge that he is likely by such act to cause death, commits the offence of culpable homicide.

Section 300. Murder
Except in the cases hereinafter excepted, culpable homicide is murder, if the act by which the death is caused is done with the intention of causing death.

Section 302. Punishment for murder
Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.

Section 375. Rape
A man is said to commit "rape" who, except in the case hereinafter excepted, has sexual intercourse with a woman under circumstances falling under any of the six following descriptions:
First.—Against her will.
Secondly.—Without her consent.
Thirdly.—With her consent, when her consent has been obtained by putting her or any person in whom she is interested in fear of death or of hurt.

Section 376. Punishment for rape
Whoever commits rape shall be punished with rigorous imprisonment of either description for a term which shall not be less than seven years, but which may extend to imprisonment for life, and shall also be liable to fine.

Section 378. Theft
Whoever, intending to take dishonestly any moveable property out of the possession of any person without that person's consent, moves that property in order to such taking, is said to commit theft.

Section 420. Cheating and dishonestly inducing delivery of property
Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine."""

    with open(statutes_dir / "indian_penal_code.txt", "w", encoding="utf-8") as f:
        f.write(ipc_content)
    
    # Code of Criminal Procedure
    crpc_content = """CODE OF CRIMINAL PROCEDURE, 1973
ACT NO. 2 OF 1974

PREAMBLE
An Act to provide for the Code of Criminal Procedure; Be it enacted by Parliament in the Twenty-fourth Year of the Republic of India as follows:

CHAPTER I - PRELIMINARY

Section 1. Short title, extent and commencement
(1) This Act may be called the Code of Criminal Procedure, 1973.
(2) It extends to the whole of India except the State of Jammu and Kashmir.

Section 2. Definitions
In this Code, unless the context otherwise requires,—
(a) "bailable offence" means an offence which is shown as bailable in the First Schedule, or which is made bailable by any other law for the time being in force; and "non-bailable offence" means any other offence;
(b) "cognizable offence" means an offence for which, and "cognizable case" means a case in which, a police officer may, in accordance with the First Schedule or under any other law for the time being in force, arrest without warrant;
(c) "complaint" means any allegation made orally or in writing to a Magistrate, with a view to his taking action under this Code, that some person, whether known or unknown, has committed an offence, but does not include a police report;

CHAPTER V - PROCEEDINGS IN PROSECUTIONS

Section 154. Information in cognizable cases
(1) Every information relating to the commission of a cognizable offence, if given orally to an officer in charge of a police station, shall be reduced to writing by him or under his direction, and be read over to the informant; and every such information, whether given in writing or reduced to writing as aforesaid, shall be signed by the person giving it, and the substance thereof shall be entered in a book to be kept by such officer in such form as the State Government may prescribe in this behalf.

Section 161. Examination of witnesses by police
(1) Any police officer making an investigation under this Chapter, or any police officer not below such rank as the State Government may, by general or special order, prescribe in this behalf, acting on the requisition of such officer, may examine orally any person supposed to be acquainted with the facts and circumstances of the case.

Section 173. Report of police officer on completion of investigation
(1) Every investigation under this Chapter shall be completed without unnecessary delay.
(2) As soon as it is completed, the officer in charge of the police station shall forward to a Magistrate empowered to take cognizance of the offence on a police report, a report in the form prescribed by the State Government.

CHAPTER IX - ORDER FOR MAINTENANCE OF WIVES, CHILDREN AND PARENTS

Section 125. Order for maintenance of wives, children and parents
(1) If any person having sufficient means neglects or refuses to maintain—
(a) his wife, unable to maintain herself, or
(b) his legitimate or illegitimate minor child, whether married or not, unable to maintain itself,
a Magistrate of the first class may, upon proof of such neglect or refusal, order such person to make a monthly allowance for the maintenance of his wife or such child, at such monthly rate not exceeding ten thousand rupees in the aggregate, as such Magistrate thinks fit."""

    with open(statutes_dir / "code_of_criminal_procedure.txt", "w", encoding="utf-8") as f:
        f.write(crpc_content)
    
    # RTI Act
    rti_content = """RIGHT TO INFORMATION ACT, 2005
ACT NO. 22 OF 2005

PREAMBLE
An Act to provide for setting out the practical regime of right to information for citizens to secure access to information under the control of public authorities, in order to promote transparency and accountability in the working of every public authority.

WHEREAS the Constitution of India has established democratic Republic;
AND WHEREAS democracy requires an informed citizenry and transparency of information which are vital to its functioning and also to contain corruption and to hold Governments and their instrumentalities accountable to the governed;

CHAPTER I - PRELIMINARY

Section 1. Short title, extent and commencement
(1) This Act may be called the Right to Information Act, 2005.
(2) It extends to the whole of India except the State of Jammu and Kashmir.

Section 2. Definitions
In this Act, unless the context otherwise requires,—
(e) "information" means any material in any form, including records, documents, memos, e-mails, opinions, advices, press releases, circulars, orders, logbooks, contracts, reports, papers, samples, models, data material held in any electronic form and information relating to any private body which can be accessed by a public authority under any other law for the time being in force;
(g) "public authority" means any authority or body or institution of self-government established or constituted by or under the Constitution;
(i) "right to information" means the right to information accessible under this Act which is held by or under the control of any public authority;

CHAPTER II - RIGHT TO INFORMATION AND OBLIGATIONS OF PUBLIC AUTHORITIES

Section 3. Right to information
Subject to the provisions of this Act, all citizens shall have the right to information.

Section 4. Obligations of public authorities
(1) Every public authority shall—
(a) maintain all its records duly catalogued and indexed in a manner and the form which facilitates the right to information under this Act;
(b) publish within one hundred and twenty days from the enactment of this Act, the particulars of its organisation, functions and duties;

Section 6. Request for obtaining information
(1) A person, who desires to obtain any information under this Act, shall make a request in writing or through electronic means in English or Hindi or in the official language of the area in which the application is being made, accompanying such fee as may be prescribed, to the Public Information Officer of the concerned public authority.

Section 7. Disposal of request
(1) The Public Information Officer shall, as expeditiously as possible and in any case within thirty days of the receipt of the request, either provide the information on payment of such fee as may be prescribed or reject the request for any of the reasons specified in sections 8 and 9."""

    with open(statutes_dir / "rti_act.txt", "w", encoding="utf-8") as f:
        f.write(rti_content)
    
    # Regulations
    regulations_dir = base_dir / "regulations"
    
    # SEBI Regulations
    sebi_content = """SECURITIES AND EXCHANGE BOARD OF INDIA (SEBI) REGULATIONS
SEBI (LISTING OBLIGATIONS AND DISCLOSURE REQUIREMENTS) REGULATIONS, 2015

PREAMBLE
In exercise of the powers conferred by section 30 of the Securities and Exchange Board of India Act, 1992, read with sub-section (1) of section 11A of the Securities Contracts (Regulation) Act, 1956, the Board hereby makes the following regulations.

CHAPTER I - PRELIMINARY

Regulation 1. Short title and commencement
(1) These regulations may be called the Securities and Exchange Board of India (Listing Obligations and Disclosure Requirements) Regulations, 2015.
(2) These regulations shall come into force on the 1st day of December, 2015.

Regulation 2. Definitions
(1) In these regulations, unless the context otherwise requires,-
(c) "listed entity" means an entity whose securities are listed on a recognised stock exchange;
(d) "market capitalisation" means the market value of the total issued shares of the listed entity as per the closing price on the recognised stock exchange;
(f) "promoter" and "promoter group" shall have the same meaning as defined in the Securities and Exchange Board of India (Issue of Capital and Disclosure Requirements) Regulations, 2018;

CHAPTER III - BOARD OF DIRECTORS AND COMMITTEES

Regulation 15. Composition of board of directors
(1) The board of directors of every listed entity shall have an optimum combination of executive and non-executive directors with at least one woman director and not less than fifty percent of the board of directors comprising non-executive directors.

Regulation 18. Audit committee
(1) Listed entity shall constitute an audit committee and such committee shall consist of a minimum of three directors with independent directors forming a majority.
(2) The chairperson of audit committee shall be an independent director.
(3) All members of audit committee shall be financially literate and at least one member shall have accounting or related financial management expertise.

CHAPTER V - RELATED PARTY TRANSACTIONS

Regulation 23. Related party transactions
(1) All related party transactions shall require prior approval of the audit committee.
(2) All material related party transactions shall require approval of the shareholders through resolution and the related parties shall not vote to approve such resolutions.
(3) A transaction with a related party shall be considered material if the transaction exceeds ten percent of the annual consolidated turnover of the listed entity.

CHAPTER VI - DISCLOSURE REQUIREMENTS

Regulation 29. Disclosure of events or information
The listed entity shall first disclose to stock exchange(s) of all events, changes or information as specified in Part A of Schedule III as soon as reasonably possible but not later than twenty four hours from the occurrence of the event or information.

CHAPTER VII - FINANCIAL RESULTS

Regulation 33. Financial results
(1) The listed entity shall submit to the stock exchange within forty five days of the end of each quarter other than the last quarter and within sixty days of the end of the last quarter, the financial results in the formats specified.

CHAPTER XII - PENALTIES

Regulation 58. Penalty for non-compliance
(1) Without prejudice to action under the Act, SEBI Act, 1992 or any other law for the time being in force, if a listed entity fails to comply with any of the provisions of these regulations, it shall be liable to penalty as prescribed under regulation 59.

Regulation 59. Penalty
For any violation of the provisions of these regulations, the recognised stock exchange shall levy fines/penalty as may be prescribed by the recognised stock exchange from time to time and approved by the Board:
Provided that the quantum of penalty shall not exceed rupees one crore per violation."""

    with open(regulations_dir / "sebi_listing_regulations.txt", "w", encoding="utf-8") as f:
        f.write(sebi_content)
    
    print("✓ Created all legal documents:")
    print(f"  - Case law: {len(list(case_law_dir.glob('*.txt')))} files")
    print(f"  - Statutes: {len(list(statutes_dir.glob('*.txt')))} files") 
    print(f"  - Regulations: {len(list(regulations_dir.glob('*.txt')))} files")

def main():
    """Main setup function"""
    
    print("🏛️ Legal Documents Setup")
    print("=" * 50)
    
    # Create folder structure
    base_dir = create_legal_document_structure()
    
    # Create all legal documents
    create_legal_documents(base_dir)
    
    print("\n" + "=" * 50)
    print("✅ Legal document setup completed!")
    print(f"📁 Documents created in: {base_dir.absolute()}")
    print("\nFolder structure:")
    print("legal_documents/")
    print("├── case_law/")
    print("│   ├── kesavananda_bharati_case.txt")
    print("│   ├── maneka_gandhi_case.txt")
    print("│   └── vishaka_case.txt")
    print("├── statutes/")
    print("│   ├── constitution_of_india.txt")
    print("│   ├── indian_penal_code.txt") 
    print("│   ├── code_of_criminal_procedure.txt")
    print("│   └── rti_act.txt")
    print("└── regulations/")
    print("    └── sebi_listing_regulations.txt")
    
    print("\n🚀 Your legal framework is ready!")
    print("Next steps:")
    print("1. Setup Groq API key in .env file")
    print("2. Run: python main.py")
    print("3. Access the API at: http://localhost:8000")

if __name__ == "__main__":
    main()