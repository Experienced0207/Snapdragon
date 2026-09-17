import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Shell } from "./components/layout/Shell";
import { LiveTranslation } from "./pages/LiveTranslation";
import { History } from "./pages/History";
import { Telemetry } from "./pages/Telemetry";
import { Settings } from "./pages/Settings";

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Shell />}>
          <Route index element={<LiveTranslation />} />
          <Route path="history" element={<History />} />
          <Route path="telemetry" element={<Telemetry />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
