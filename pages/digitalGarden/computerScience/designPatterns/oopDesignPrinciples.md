# OOP Design Principles

## SOLID

### S - Single Responsibility Principle

A class/method should only have a single purpose/responsibility and therfore only one reason to change.

### O - Open/Closed Principle

Classes should be open for extension but closed for modification. To extend the behavior, new code should be added however old code should not have to be modified. This then prevents situations in which a change to classes also requires adaption of all depending classes. This is achieved with interfaces which allow different implementations by keep the same API.

### L - Liskov Substitution Principle

Subtypes should be substitutable for their base types.

### I - Interface Segregation Principle

Make fine grained interfaces that are client specific instead of general purpose interfaces. (Which slightly contradicts the strategy pattern)

### D - Dependency inversion Principle

Depend on Abstractions not on concrete classes etc.

## Favor Compositon over Inheritance

By using composition for example in the strategy pattern instead of inheritance it allows us to be flexible at runtime.

## Program to an interface, not and implementation

Avoid referencing concrete classes, declare interfaces instead as then implementations are easly switched out.

## Encapsulate what varies

By encapsulating/hiding the parts that can vary for example implementations behind interfaces we can minimize the impact of that code because thanks to the interface we have a unified API.
