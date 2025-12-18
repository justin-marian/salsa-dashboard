import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./style_main.css";

// Locate the root DOM node defined in index.html
const container = document.getElementById("root");
// Create a React 18 concurrent root
const root = createRoot(container);
// Mount the full application
root.render(<App />);
