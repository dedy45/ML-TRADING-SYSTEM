//+------------------------------------------------------------------+
//| Fibonacci Deep Learning EA Integration                           |
//| Template untuk integrasi dengan Python Deep Learning Model      |
//+------------------------------------------------------------------+

#property copyright "Fibonacci Deep Learning EA"
#property version   "1.00"

//--- Input parameters
input string PythonSignalFile = "fibonacci_signal.json";      // Path ke signal file
input string PythonRequestFile = "fibonacci_request.json";    // Path ke request file
input double MinConfidence = 0.70;                            // Minimum confidence untuk execute
input double RiskPerTrade = 2.0;                             // Risk per trade (%)
input bool   UsePythonSignals = true;                        // Enable Python signals
input int    SignalValiditySeconds = 300;                    // Signal validity (5 minutes)

//--- Global variables
datetime lastSignalTime = 0;
string currentSignalType = "HOLD";
double currentConfidence = 0.0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("🧠 Fibonacci Deep Learning EA started");
    Print("📁 Signal file: ", PythonSignalFile);
    Print("📁 Request file: ", PythonRequestFile);
    Print("🎯 Min confidence: ", MinConfidence);
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    if (!UsePythonSignals) return;
    
    // Check untuk new signal setiap 30 detik
    static datetime lastCheck = 0;
    if (TimeCurrent() - lastCheck < 30) return;
    lastCheck = TimeCurrent();
    
    // Send request ke Python model
    SendRequestToPython();
    
    // Read signal dari Python
    if (ReadSignalFromPython())
    {
        ProcessPythonSignal();
    }
}

//+------------------------------------------------------------------+
//| Send market data request ke Python model                        |
//+------------------------------------------------------------------+
void SendRequestToPython()
{
    // Get current market data
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double fibLevel = CalculateFibonacciLevel();  // Your existing Fib calculation
    
    // Create request JSON
    string request = StringFormat(
        "{\n"
        "  \"symbol\": \"%s\",\n"
        "  \"timeframe\": \"%s\",\n"
        "  \"current_price\": %.5f,\n"
        "  \"fibonacci_level\": %.1f,\n"
        "  \"session_asia\": %d,\n"
        "  \"session_europe\": %d,\n"
        "  \"session_us\": %d,\n"
        "  \"tp\": %.5f,\n"
        "  \"sl\": %.5f,\n"
        "  \"hour\": %d,\n"
        "  \"request_time\": \"%s\"\n"
        "}",
        _Symbol,
        PeriodToString(_Period),
        currentPrice,
        fibLevel,
        IsAsiaSession() ? 1 : 0,
        IsEuropeSession() ? 1 : 0,
        IsUSSession() ? 1 : 0,
        CalculateTP(currentPrice, fibLevel),
        CalculateSL(currentPrice, fibLevel),
        TimeHour(TimeCurrent()),
        TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS)
    );
    
    // Write request file
    int handle = FileOpen(PythonRequestFile, FILE_WRITE|FILE_TXT);
    if (handle != INVALID_HANDLE)
    {
        FileWriteString(handle, request);
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| Read signal dari Python model                                   |
//+------------------------------------------------------------------+
bool ReadSignalFromPython()
{
    if (!FileIsExist(PythonSignalFile)) return false;
    
    int handle = FileOpen(PythonSignalFile, FILE_READ|FILE_TXT);
    if (handle == INVALID_HANDLE) return false;
    
    string signalJson = "";
    while (!FileIsEnding(handle))
    {
        signalJson += FileReadString(handle);
    }
    FileClose(handle);
    
    // Parse JSON signal (simplified parsing)
    return ParseSignalJSON(signalJson);
}

//+------------------------------------------------------------------+
//| Parse JSON signal dari Python                                   |
//+------------------------------------------------------------------+
bool ParseSignalJSON(string json)
{
    // Simplified JSON parsing - extract key values
    // In production, use proper JSON library
    
    // Extract signal_type
    int pos = StringFind(json, "\"signal_type\":");
    if (pos >= 0)
    {
        string temp = StringSubstr(json, pos);
        pos = StringFind(temp, "\"");
        if (pos >= 0)
        {
            temp = StringSubstr(temp, pos + 1);
            pos = StringFind(temp, "\"");
            if (pos >= 0)
            {
                currentSignalType = StringSubstr(temp, 0, pos);
            }
        }
    }
    
    // Extract confidence
    pos = StringFind(json, "\"confidence\":");
    if (pos >= 0)
    {
        string temp = StringSubstr(json, pos + 12);
        pos = StringFind(temp, ",");
        if (pos >= 0)
        {
            temp = StringSubstr(temp, 0, pos);
            currentConfidence = StringToDouble(temp);
        }
    }
    
    // Extract timestamp untuk validity check
    pos = StringFind(json, "\"timestamp\":");
    if (pos >= 0)
    {
        // Parse timestamp and check validity
        lastSignalTime = TimeCurrent(); // Simplified
    }
    
    return (currentSignalType != "" && currentConfidence > 0);
}

//+------------------------------------------------------------------+
//| Process signal dari Python model                                |
//+------------------------------------------------------------------+
void ProcessPythonSignal()
{
    // Check signal validity (time-based)
    if (TimeCurrent() - lastSignalTime > SignalValiditySeconds)
    {
        Print("⚠️ Signal expired");
        return;
    }
    
    // Check confidence threshold
    if (currentConfidence < MinConfidence)
    {
        Print("📊 Signal confidence too low: ", currentConfidence);
        return;
    }
    
    // Execute signal
    if (currentSignalType == "BUY")
    {
        ExecuteBuySignal();
    }
    else if (currentSignalType == "SELL")
    {
        ExecuteSellSignal();
    }
    
    Print("🧠 Python signal processed: ", currentSignalType, 
          " (confidence: ", currentConfidence * 100, "%)");
}

//+------------------------------------------------------------------+
//| Execute BUY signal                                               |
//+------------------------------------------------------------------+
void ExecuteBuySignal()
{
    // Check existing positions
    if (PositionsTotal() > 0) return;
    
    double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double sl = CalculateSL(price, 0);  // Your SL calculation
    double tp = CalculateTP(price, 0);  // Your TP calculation
    double lot = CalculateLotSize(RiskPerTrade, MathAbs(price - sl));
    
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = lot;
    request.type = ORDER_TYPE_BUY;
    request.price = price;
    request.sl = sl;
    request.tp = tp;
    request.comment = StringFormat("DL_Fib_%.0f%%", currentConfidence * 100);
    
    if (OrderSend(request, result))
    {
        Print("✅ BUY order opened: ", result.order, 
              " | Confidence: ", currentConfidence * 100, "%");
    }
    else
    {
        Print("❌ BUY order failed: ", result.retcode);
    }
}

//+------------------------------------------------------------------+
//| Execute SELL signal                                              |
//+------------------------------------------------------------------+
void ExecuteSellSignal()
{
    // Similar to ExecuteBuySignal but for SELL
    // Implementation follows same pattern
    Print("📉 SELL signal processing...");
    // Add your SELL logic here
}

//+------------------------------------------------------------------+
//| Calculate Fibonacci level (your existing logic)                 |
//+------------------------------------------------------------------+
double CalculateFibonacciLevel()
{
    // Use your existing Fibonacci calculation
    // Return values like 0.0 (B_0), -1.8 (B_-1.8), etc.
    return 0.0;  // Placeholder
}

//+------------------------------------------------------------------+
//| Calculate Take Profit                                            |
//+------------------------------------------------------------------+
double CalculateTP(double price, double fibLevel)
{
    // Your existing TP calculation
    // Maintain 2:1 ratio untuk optimal performance
    return price + (price * 0.001); // Placeholder
}

//+------------------------------------------------------------------+
//| Calculate Stop Loss                                              |
//+------------------------------------------------------------------+
double CalculateSL(double price, double fibLevel)
{
    // Your existing SL calculation
    return price - (price * 0.0005); // Placeholder
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                 |
//+------------------------------------------------------------------+
double CalculateLotSize(double riskPercent, double slDistance)
{
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * riskPercent / 100.0;
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    
    double slInTicks = slDistance / tickSize;
    double lotSize = riskAmount / (slInTicks * tickValue);
    
    // Normalize lot size
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    lotSize = MathMax(minLot, MathMin(maxLot, 
              MathRound(lotSize / lotStep) * lotStep));
    
    return lotSize;
}

//+------------------------------------------------------------------+
//| Check if Asia session is active                                 |
//+------------------------------------------------------------------+
bool IsAsiaSession()
{
    int hour = TimeHour(TimeCurrent());
    return (hour >= 23 || hour < 8);  // 23:00-08:00 GMT
}

//+------------------------------------------------------------------+
//| Check if Europe session is active                               |
//+------------------------------------------------------------------+
bool IsEuropeSession()
{
    int hour = TimeHour(TimeCurrent());
    return (hour >= 8 && hour < 16);  // 08:00-16:00 GMT
}

//+------------------------------------------------------------------+
//| Check if US session is active                                   |
//+------------------------------------------------------------------+
bool IsUSSession()
{
    int hour = TimeHour(TimeCurrent());
    return (hour >= 13 && hour < 21);  // 13:00-21:00 GMT
}

//+------------------------------------------------------------------+
//| Convert period to string                                         |
//+------------------------------------------------------------------+
string PeriodToString(ENUM_TIMEFRAMES period)
{
    switch(period)
    {
        case PERIOD_M1:  return "M1";
        case PERIOD_M5:  return "M5";
        case PERIOD_M15: return "M15";
        case PERIOD_M30: return "M30";
        case PERIOD_H1:  return "H1";
        case PERIOD_H4:  return "H4";
        case PERIOD_D1:  return "D1";
        default:         return "M15";
    }
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("🛑 Fibonacci Deep Learning EA stopped");
    
    // Cleanup files
    if (FileIsExist(PythonSignalFile))
        FileDelete(PythonSignalFile);
    if (FileIsExist(PythonRequestFile))
        FileDelete(PythonRequestFile);
}
