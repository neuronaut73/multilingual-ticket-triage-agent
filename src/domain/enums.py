from enum import Enum


class Topic(str, Enum):
    POLICY    = "Policy / Contract"
    CLAIMS    = "Claims / Damage"
    BILLING   = "Billing / Payment"
    TECHNICAL = "Technical / Online Access"
    OTHER     = "Other"


class Urgency(str, Enum):
    LOW    = "Low"
    MEDIUM = "Medium"
    HIGH   = "High"


class NextAction(str, Enum):
    SEND_FAQ          = "send_standard_faq_or_self_service_link"
    CREATE_CLAIM      = "create_or_update_claim"
    FORWARD_BILLING   = "forward_to_billing_team"
    FORWARD_TECHNICAL = "forward_to_technical_support"
    ESCALATE          = "escalate_to_human_supervisor"
    ASK_MORE_INFO     = "ask_for_more_information"
