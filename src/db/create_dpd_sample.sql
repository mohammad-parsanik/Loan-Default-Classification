IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'D_ANALYTICS')
BEGIN
    EXEC('CREATE SCHEMA [D_ANALYTICS]');
END
GO

IF OBJECT_ID('D_ANALYTICS.DPD_SAMPLE', 'U') IS NOT NULL
BEGIN
    DROP TABLE [D_ANALYTICS].[DPD_SAMPLE];
END
GO

CREATE TABLE [D_ANALYTICS].[DPD_SAMPLE]
(
  [LOAN_ID]                       BIGINT,
  [SNAPSHOT_DATE]                 BIGINT,
  [DPD_DAYS]                      INT,
  [LOAN_CATEGORY]                 INT,
  [DAYS_TO_NEXT_THRESHOLD]        INT,
  [PAYED_OVERDUE_INST_CNT]        INT,
  [UNPAYED_INST_CNT]              INT,
  [PAYED_OVERDUE_AMNT]            DECIMAL(18, 2),
  [MATURED_INST_CNT]              INT,
  [OVERDUE_RATIO]                 FLOAT,
  [IS_IN_WARNING_ZONE]            INT,
  [ONTIME_RATIO]                  FLOAT,
  [UPCOMING_INST_CNT]             INT,
  [UPCOMING_AMNT]                 DECIMAL(18, 2),
  [CNT_INSTALLMENT_WARNING_ZONE]  INT,
  [DPD_DAYS_T1]                   INT,
  [DPD_DAYS_T2]                   INT,
  [DPD_DAYS_T3]                   INT,
  [DPD_DAYS_T4]                   INT,
  [DPD_DAYS_T5]                   INT,
  [CATEGORY_T1]                   INT,
  [CATEGORY_T2]                   INT,
  [CATEGORY_T3]                   INT,
  [DPD_TREND_1M]                  FLOAT,
  [DPD_TREND_3M]                  FLOAT,
  [CATEGORY_TREND_1M]             FLOAT,
  [CATEGORY_TREND_3M]             FLOAT,
  [IS_DETERIORATING]              INT,
  [IS_ACCELERATING]               INT,
  [IS_IMPROVING]                  INT,
  [MONTHS_IN_CURRENT_CATEGORY]    INT,
  [HIST_MAX_DPD_DAYS]             INT,
  [HIST_MAX_CATEGORY]             INT,
  [HAS_EVER_BEEN_NPL]             INT,
  [HAS_EVER_BEEN_PRENPL]          INT,
  [HAS_RECOVERED_BEFORE]          INT,
  [CNT_RECOVERED_BEFORE]          INT,
  [COUNT_CATEGORY_CHANGES]        INT,
  [COUNT_DPD_EVENTS_LAST_3M]      INT,
  [COUNT_DPD_EVENTS_LAST_6M]      INT,
  [COUNT_30PLUS_DPD_LAST_3M]      INT,
  [COUNT_60PLUS_DPD_LAST_3M]      INT,
  [COUNT_90PLUS_DPD_LAST_3M]      INT,
  [MAX_DPD_LAST_3M]               INT,
  [MAX_DPD_LAST_6M]               INT,
  [TOTAL_DPD_DAYS_LAST_3M]        INT,
  [TOTAL_DPD_DAYS_LAST_6M]        INT,
  [CONSECUTIVE_MONTHS_WITH_DPD]   INT,
  [DAYS_SINCE_LAST_DPD]           INT,
  [DAYS_SINCE_LAST_30_DPD]        INT,
  [DAYS_SINCE_LAST_60_DPD]        INT,
  [DAYS_SINCE_LAST_90_DPD]        INT,
  [WORST_CLOSED_LOAN_DPD]         INT,
  [AVERAGE_CLOSE_LOAN_DPD]        FLOAT,
  [MAX_DPD_ANY_PAST_LOAN]         INT,
  [AVG_DPD_OTHER_LOANS]           FLOAT,
  [PRE_UPTO30_DPD_LOANS]          INT,
  [PRE_UPTO60_DPD_LOANS]          INT,
  [PRE_UPTO120_DPD_LOANS]         INT,
  [PRE_UPTO150_DPD_LOANS]         INT,
  [COUNT_ACTIVE_CONTRACTS]        INT,
  [COUNT_DELINQUENT_CONTRACTS]    INT,
  [CONTRACT_AGE_MONTH]            INT,
  [PCT_COMPLETED]                 FLOAT,
  [REMAINING_INST_CNT]            INT,
  [REMAINING_AMNT]                DECIMAL(18, 2),
  [WORST_FUTURE_DPD]              INT,
  [WORST_FUTURE_CAT]              INT,
  [NATIONAL_CODE]                 VARCHAR(20),
  [CONTRACT_NUMBER]               VARCHAR(50)
)
GO
