import React from "react";
import { types } from "../../editor/components";
import NumericInput from "./NumericInput";
import Vector3Input from "./Vector3Input";
import EulerInput from "./EulerInput";
import ColorInput from "./ColorInput";
import FileInput from "./FileInput";
import BooleanInput from "./BooleanInput";

/* eslint react/display-name: 0 */
const componentTypeMappings = new Map([
  [types.color, (value, onChange) => <ColorInput value={value} onChange={onChange} />],
  [
    types.number,
    (value, onChange, { min, max }) => <NumericInput value={value} onChange={onChange} min={min} max={max} />
  ],
  [types.vector, (value, onChange) => <Vector3Input value={value} onChange={onChange} />],
  [types.euler, (value, onChange) => <EulerInput value={value} onChange={onChange} />],
  [types.boolean, (value, onChange) => <BooleanInput value={value} onChange={onChange} />],
  [types.file, (value, onChange, { filters }) => <FileInput {...{ value, onChange, filters }} />]
]);

export default componentTypeMappings;
